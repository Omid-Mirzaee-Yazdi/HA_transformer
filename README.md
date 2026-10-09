# Home World Model

A local Home Assistant listener, temporal action model, live dashboard, and read-only HA sensor integration. The service listens to the HA event bus and publishes model state; it does not call device services.

## Run locally

Use a Python environment with Apple Silicon PyTorch/MPS support on an M-series Mac:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ha_world_model.server
```

Open `http://127.0.0.1:8765`. Enter the Home Assistant base URL, a long-lived access token, and the HA IANA timezone. The server then trains from `data/ha_history.sqlite3` using that timezone and saves `data/world_model.pt`. The token is stored locally in `data/server_config.json` with owner-only permissions; it is never returned by the API or written to logs.

For HTTPS installations using a private certificate authority, enter the local PEM CA certificate path in the optional setup field. The server adds that CA to the default trust store for both the event WebSocket and REST requests; certificate verification is never disabled. Publicly trusted certificates need no extra path.

The service binds to the Mac's network interfaces so a separate Home Assistant host can reach its reporting endpoint. Setup and dashboard APIs accept only loopback requests. The HA integration endpoint requires a separate generated bridge token. Allow port `8765` only on your trusted home network; the event listener uses the HA access token to subscribe to all event types and read initial states.

## Home Assistant sensors

For a HACS custom-repository install, make this GitHub repository public, then add its URL in HACS > Integrations > menu > Custom repositories as an **Integration**. Download **Home World Model**, restart Home Assistant, then add it from Settings > Devices & Services. Use the server URL and bridge token shown under the dashboard's settings. HACS installs only the integration; the model server must remain running on the Mac.

Alternatively, copy `custom_components/world_model` into the Home Assistant configuration directory's `custom_components` folder and restart. The integration polls the server's read-only endpoint and creates status, activity, next-action, action-score, and event-count sensors.

## Model boundaries

The model uses a rolling 60-minute sequence and ranks recurring direct-user action candidates. Current action classes are sparse and scores are not calibrated probabilities. The sensor reports recommendations only; there is deliberately no Home Assistant service-call path. The two protected May backups must be decrypted locally before they can contribute to the canonical history.