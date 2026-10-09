"""Live Home Assistant event ingestion and sequence inference."""

import asyncio
import json
import os
import secrets
import socket
import ssl
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import websockets
import certifi

from ha_world_model.model import SEQUENCE_LENGTH, load_model, train_model
import world_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_PATH = DATA_DIR / "server_config.json"
MODEL_PATH = DATA_DIR / "world_model.pt"
ACTION_LOOKBACK_SECONDS = (3600, 86400, 604800)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class LiveRuntime:
    def __init__(self):
        self.config = self._load_config()
        self.config.setdefault("bridge_token", secrets.token_urlsafe(32))
        self.connected = False
        self.connection_state = "needs_setup"
        self.connection_error = None
        self.model = None
        self.model_status = "needs_setup"
        self.model_summary = None
        self.training_error = None
        self.states = {}
        self.snapshots = deque(maxlen=SEQUENCE_LENGTH)
        self.action_times = deque(maxlen=2000)
        self.action_contexts = deque(maxlen=2000)
        self.recent_events = deque(maxlen=30)
        self.event_types = Counter()
        self.event_count = 0
        self.last_event_at = None
        self.last_event_type = None
        self.last_snapshot_bucket = None
        self.last_inference_at = None
        self.prediction = {"action_score": None, "candidates": []}
        self._connection_task = None
        self._training_task = None
        self._sampling_task = None
        self._stopping = False
        self._write_config()

    @staticmethod
    def _load_config():
        if not CONFIG_PATH.exists():
            return {"ha_url": "http://homeassistant.local:8123", "access_token": "",
                    "timezone": "UTC"}
        try:
            return json.loads(CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return {"ha_url": "http://homeassistant.local:8123", "access_token": "",
                    "timezone": "UTC"}

    def _write_config(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        temporary = CONFIG_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.config, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        temporary.replace(CONFIG_PATH)

    async def start(self):
        self._sampling_task = asyncio.create_task(self._snapshot_loop())
        if self.config.get("ha_url") and self.config.get("access_token"):
            self._training_task = asyncio.create_task(self.retrain())
            self._start_connection()

    async def stop(self):
        self._stopping = True
        for task in (self._connection_task, self._training_task, self._sampling_task):
            if task:
                task.cancel()
        await asyncio.gather(*(task for task in (self._connection_task, self._training_task,
                                                  self._sampling_task) if task),
                             return_exceptions=True)

    async def _snapshot_loop(self):
        while not self._stopping:
            if self.connected and self.model:
                self._take_snapshot(time.time())
            await asyncio.sleep(5)

    async def retrain(self):
        self.model_status = "training"
        self.training_error = None
        try:
            model, summary = await asyncio.to_thread(
                train_model, ROOT / "data" / "ha_history.sqlite3", MODEL_PATH,
                self.config.get("timezone", "UTC"), 24)
            self.model = model
            self.model_summary = summary
            self.model_status = "ready"
            self.snapshots.clear()
            self.last_snapshot_bucket = None
        except Exception as error:
            self.model_status = "error"
            self.training_error = str(error)

    def update_config(self, ha_url, access_token, timezone_name, ca_cert_path=None):
        timezone_changed = self.config.get("timezone", "UTC") != (timezone_name or "UTC")
        self.config.update({"ha_url": ha_url.rstrip("/"), "access_token": access_token,
                            "timezone": timezone_name or "UTC"})
        if ca_cert_path is not None:
            self.config["ca_cert_path"] = ca_cert_path
        self._write_config()
        if self.model is None or timezone_changed:
            if not self._training_task or self._training_task.done():
                self._training_task = asyncio.create_task(self.retrain())
        self._start_connection()

    def _start_connection(self):
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
        self._connection_task = asyncio.create_task(self._connect_loop())

    def _websocket_url(self):
        base = self.config["ha_url"].rstrip("/")
        if base.startswith("https://"):
            return "wss://" + base[len("https://"): ] + "/api/websocket"
        if base.startswith("http://"):
            return "ws://" + base[len("http://"): ] + "/api/websocket"
        raise ValueError("Home Assistant URL must start with http:// or https://")

    def _ssl_context(self):
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=certifi.where())
        ca_cert_path = self.config.get("ca_cert_path")
        if ca_cert_path:
            context.load_verify_locations(cafile=str(Path(ca_cert_path).expanduser()))
        return context

    async def _connect_loop(self):
        delay = 2
        while not self._stopping:
            try:
                self.connection_state = "connecting"
                self.connection_error = None
                websocket_url = self._websocket_url()
                options = {"ssl": self._ssl_context()} if websocket_url.startswith("wss://") else {}
                async with websockets.connect(websocket_url, ping_interval=20,
                                              open_timeout=15, max_size=2**23,
                                              **options) as socket:
                    greeting = json.loads(await socket.recv())
                    if greeting.get("type") != "auth_required":
                        raise RuntimeError("Home Assistant did not request WebSocket authentication")
                    await socket.send(json.dumps({"type": "auth",
                                                  "access_token": self.config["access_token"]}))
                    authenticated = json.loads(await socket.recv())
                    if authenticated.get("type") != "auth_ok":
                        raise RuntimeError(authenticated.get("message", "Home Assistant authentication failed"))
                    await socket.send(json.dumps({"id": 1, "type": "subscribe_events"}))
                    subscription = json.loads(await socket.recv())
                    if subscription.get("type") != "result" or not subscription.get("success"):
                        raise RuntimeError("Home Assistant rejected event subscription")
                    self.connected = True
                    self.connection_state = "connected"
                    delay = 2
                    await self._fetch_current_states()
                    async for raw in socket:
                        message = json.loads(raw)
                        if message.get("type") == "event":
                            self.handle_event(message.get("event", {}))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.connected = False
                self.connection_state = "reconnecting"
                self.connection_error = str(error)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
            finally:
                self.connected = False
                if not self._stopping:
                    self.connection_state = "reconnecting"

    async def _fetch_current_states(self):
        url = self.config["ha_url"].rstrip("/") + "/api/states"
        headers = {"Authorization": f"Bearer {self.config['access_token']}"}
        async with httpx.AsyncClient(timeout=15, verify=self._ssl_context()) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            for item in response.json():
                entity_id = item.get("entity_id")
                if entity_id:
                    self.states[entity_id] = {"state": item.get("state"),
                                              "attributes": item.get("attributes") or {}}
        self._take_snapshot(time.time(), force=True)

    def handle_event(self, event):
        event_type = event.get("event_type", "unknown")
        fired = event.get("time_fired")
        try:
            fired_seconds = datetime.fromisoformat(str(fired).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            fired_seconds = time.time()
        self.event_count += 1
        self.event_types[event_type] += 1
        self.last_event_at = utc_now()
        self.last_event_type = event_type
        data = event.get("data") or {}
        summary = {"time": self.last_event_at, "type": event_type}
        if event_type == "state_changed":
            state = data.get("new_state")
            if isinstance(state, dict) and state.get("entity_id"):
                entity_id = state["entity_id"]
                self.states[entity_id] = {"state": state.get("state"),
                                          "attributes": state.get("attributes") or {}}
                summary["entity"] = entity_id
                summary["state"] = state.get("state")
        elif event_type == "call_service":
            context = event.get("context") or {}
            label = self._service_label(data)
            summary["action"] = label
            if context.get("user_id") and not context.get("parent_id") and label:
                context_id = context.get("id")
                if context_id not in self.action_contexts:
                    self.action_contexts.append(context_id)
                    self.action_times.append(fired_seconds)
        self.recent_events.appendleft(summary)
        self._take_snapshot(fired_seconds)

    @staticmethod
    def _service_label(data):
        domain, service = data.get("domain"), data.get("service")
        service_data = data.get("service_data") or {}
        target = (data.get("target") or {}).get("entity_id") or service_data.get("entity_id")
        if isinstance(target, dict):
            target = target.get("entity_id")
        if isinstance(target, list):
            target = ",".join(str(item) for item in target)
        return f"{domain}.{service} -> {target}" if domain and service and target else None

    def _take_snapshot(self, timestamp, force=False):
        if not self.model:
            return
        bucket = int(timestamp // 300)
        if not force and bucket == self.last_snapshot_bucket:
            return
        self.last_snapshot_bucket = bucket
        row = self._feature_snapshot(timestamp)
        if row is None:
            return
        self.snapshots.append(row)
        self.last_inference_at = utc_now()
        if len(self.snapshots) < SEQUENCE_LENGTH:
            return
        self.prediction = self.model.predict(np.stack(self.snapshots))

    def _feature_snapshot(self, timestamp):
        if not self.model:
            return None
        current = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(
            __import__("zoneinfo").ZoneInfo(self.config.get("timezone", "UTC")))
        minute = current.hour * 60 + current.minute
        week_minute = current.weekday() * 1440 + minute
        cyclical = {
            "tod_sin": np.sin(2 * np.pi * minute / 1440),
            "tod_cos": np.cos(2 * np.pi * minute / 1440),
            "week_sin": np.sin(2 * np.pi * week_minute / 10080),
            "week_cos": np.cos(2 * np.pi * week_minute / 10080),
            "weekend": float(current.weekday() >= 5),
        }
        action_times = np.asarray(self.action_times, dtype=np.float64)
        values = []
        for feature in self.model.feature_columns:
            if feature in cyclical:
                values.append(cyclical[feature])
                continue
            if feature.startswith("user_actions_"):
                lookbacks = {"user_actions_1h": 3600, "user_actions_24h": 86400,
                             "user_actions_7d": 604800}
                lookback = lookbacks[feature]
                values.append(int(np.sum((action_times >= timestamp - lookback) & (action_times < timestamp))))
                continue
            if feature == "minutes_since_user_action":
                values.append((timestamp - action_times[-1]) / 60 if len(action_times) else np.nan)
                continue
            value = self._state_feature(feature)
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def _state_feature(self, feature):
        state_feature = feature[:-5] if feature.endswith("_lag0") else feature
        if state_feature in self.states:
            return self._numeric_or_binary(state_feature, self.states[state_feature].get("state"))
        entities = sorted(self.states, key=len, reverse=True)
        for entity_id in entities:
            item = self.states[entity_id]
            if state_feature.startswith(entity_id + "."):
                attribute = state_feature[len(entity_id) + 1:]
                value = (item.get("attributes") or {}).get(attribute)
                try:
                    return float(value) if value is not None else np.nan
                except (TypeError, ValueError):
                    return np.nan
        for entity_id in entities:
            item = self.states[entity_id]
            prefix = entity_id + "_"
            if state_feature.startswith(prefix):
                category = state_feature[len(prefix):]
                return float(str(item.get("state", "")).casefold() == category.casefold())
        return np.nan

    @staticmethod
    def _numeric_or_binary(entity_id, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            domain = entity_id.partition(".")[0]
            if domain in ("binary_sensor", "person", "device_tracker", "automation",
                          "cover", "fan", "input_boolean", "light", "lock", "script", "switch"):
                return world_model.to_binary(value)
            return np.nan

    def current_activity(self):
        people_home = []
        presence_active = []
        for entity_id, item in self.states.items():
            value = str(item.get("state", "")).casefold()
            friendly = (item.get("attributes") or {}).get("friendly_name", entity_id)
            if entity_id.startswith(("person.", "device_tracker.")) and value == "home":
                people_home.append(friendly)
            if entity_id.startswith("binary_sensor.") and value in world_model.TRUE_STATES:
                lowered = entity_id.casefold()
                if any(word in lowered for word in ("presence", "occupancy", "motion")):
                    presence_active.append(friendly)
        parts = []
        if people_home:
            parts.append("Home: " + ", ".join(sorted(set(people_home))))
        if presence_active:
            parts.append("Activity detected: " + ", ".join(sorted(set(presence_active))[:4]))
        return "; ".join(parts) if parts else "No active presence signal"

    def dashboard_state(self):
        trained_at = None
        if MODEL_PATH.exists():
            try:
                trained_at = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                pass
        warmup = max(0, SEQUENCE_LENGTH - len(self.snapshots)) * 5
        integration_url = "http://<mac-lan-ip>:8765"
        parsed = urlparse(self.config.get("ha_url", ""))
        if parsed.hostname:
            try:
                address = socket.getaddrinfo(parsed.hostname, parsed.port or 8123,
                                             socket.AF_INET, socket.SOCK_DGRAM)[0][4][0]
                route = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                route.connect((address, parsed.port or 8123))
                integration_url = f"http://{route.getsockname()[0]}:8765"
                route.close()
            except OSError:
                pass
        return {
            "connected": self.connected,
            "connection_state": self.connection_state,
            "connection_error": self.connection_error,
            "configured": bool(self.config.get("access_token")),
            "ha_url": self.config.get("ha_url"),
            "timezone": self.config.get("timezone", "UTC"),
            "ca_cert_path": self.config.get("ca_cert_path", ""),
            "bridge_token": self.config.get("bridge_token"),
            "integration_url": integration_url,
            "model_status": self.model_status,
            "model_error": self.training_error,
            "model_device": self.model_summary.get("device") if self.model_summary else None,
            "model_features": self.model_summary.get("features") if self.model_summary else None,
            "model_samples": self.model_summary.get("samples") if self.model_summary else None,
            "model_trained_at": trained_at,
            "current_activity": self.current_activity(),
            "events_received": self.event_count,
            "last_event_at": self.last_event_at,
            "last_event_type": self.last_event_type,
            "recent_events": list(self.recent_events),
            "event_types": self.event_types.most_common(8),
            "warmup_minutes": warmup,
            "inference_at": self.last_inference_at,
            "prediction": self.prediction,
        }

    def ha_sensor_state(self):
        state = self.dashboard_state()
        state.pop("bridge_token", None)
        state.pop("connection_error", None)
        state.pop("model_error", None)
        state.pop("ca_cert_path", None)
        state.pop("recent_events", None)
        return state