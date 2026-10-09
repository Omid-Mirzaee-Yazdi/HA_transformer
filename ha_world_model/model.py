"""Train and run the persisted temporal intent model."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import world_model as features


NO_ACTION = "<no action>"
OTHER_ACTION = "<other action>"
MIN_ACTION_EPISODES = 8
MAX_ACTION_CLASSES = 12
SEQUENCE_LENGTH = 12


class ActionModel:
    def __init__(self, model, labels, feature_columns, medians, means, scales,
                 device, history_end, timezone_name):
        self.model = model
        self.labels = labels
        self.feature_columns = feature_columns
        self.medians = np.asarray(medians, dtype=np.float32)
        self.means = np.asarray(means, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.device = torch.device(device)
        self.history_end = history_end
        self.timezone_name = timezone_name

    def predict(self, sequence):
        values = np.asarray(sequence, dtype=np.float32)
        if values.shape != (SEQUENCE_LENGTH, len(self.feature_columns)):
            raise ValueError("Live sequence does not match the trained model shape")
        values = np.where(np.isfinite(values), values, self.medians)
        values = np.clip((values - self.means) / self.scales, -8, 8)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(values[None]).to(self.device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        ranked = sorted(zip(self.labels, probabilities), key=lambda item: item[1], reverse=True)
        action_score = float(sum(score for label, score in ranked if label != NO_ACTION))
        candidates = [
            {"action": label, "score": float(score)}
            for label, score in ranked
            if label not in (NO_ACTION, OTHER_ACTION)
        ][:5]
        return {"action_score": action_score, "candidates": candidates,
                "other_action_score": float(dict(ranked).get(OTHER_ACTION, 0.0))}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "input_size": len(self.feature_columns),
            "labels": self.labels,
            "feature_columns": self.feature_columns,
            "medians": torch.as_tensor(self.medians, dtype=torch.float32),
            "means": torch.as_tensor(self.means, dtype=torch.float32),
            "scales": torch.as_tensor(self.scales, dtype=torch.float32),
            "state_dict": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
            "history_end": self.history_end,
            "timezone": self.timezone_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, path)


def load_model(path, device=None):
    path = Path(path)
    device = device or preferred_device()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = features.TemporalActionNetwork(
        payload["input_size"], hidden_size=24, num_classes=len(payload["labels"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return ActionModel(model, payload["labels"], payload["feature_columns"],
                       payload["medians"], payload["means"], payload["scales"],
                       device, payload["history_end"], payload["timezone"])


def preferred_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _history_frames(connection, timezone_name):
    entity, join, _, matches = features.get_entities(connection)
    contextual = [row[0] for row in connection.execute(
        f"SELECT DISTINCT {entity} FROM states s{join} WHERE ({matches}) ORDER BY {entity}")]
    windows = features.history_windows(connection)
    if not contextual or not windows:
        raise RuntimeError("Canonical recorder history has no usable entities or time windows")

    window_inputs = []
    all_records = []
    for start, end in windows:
        records = features.direct_user_action_records(connection, start, end)
        all_records.extend(records)
        window_inputs.append((start, end, records))
    class_counts = {}
    for _, label in all_records:
        class_counts[label] = class_counts.get(label, 0) + 1
    action_classes = [label for label, count in
                      sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
                      if count >= MIN_ACTION_EPISODES][:MAX_ACTION_CLASSES]
    if not action_classes:
        raise RuntimeError("Not enough repeated direct user actions to train action candidates")

    first_start = min(start for start, _, _ in window_inputs)
    last_end = max(end for _, end, _ in window_inputs)
    controls = features.get_control_entities(connection, first_start, last_end)
    entities = sorted(set(contextual) | set(controls))
    frames = []
    for start, end, records in window_inputs:
        targets = {name for name in contextual if name.startswith("binary_sensor.")
               and ("presence" in name or "occupancy" in name)}
        data = features.make_features(connection, entities, targets, start, end,
                          timezone_name, require_targets=False)
        if not data:
            continue
        frame, _, _ = data
        action_times = np.asarray([time for time, _ in records], dtype=np.float64)
        history = features.action_history_features(frame, action_times)
        selected = [name for name in history.columns
                    if name.endswith("_lag0") or name.startswith("user_actions_")
                    or name == "minutes_since_user_action"]
        history = history[selected]
        grid = history.index.asi8 / 1e9
        event_times = np.asarray([time for time, _ in records], dtype=np.float64)
        event_labels = [label for _, label in records]
        next_event = np.searchsorted(event_times, grid, side="right")
        labels = np.full(len(grid), NO_ACTION, dtype=object)
        available = next_event < len(event_times)
        within_horizon = np.zeros(len(grid), dtype=bool)
        within_horizon[available] = event_times[next_event[available]] <= grid[available] + 900
        for index in np.flatnonzero(within_horizon):
            action = event_labels[next_event[index]]
            labels[index] = action if action in action_classes else OTHER_ACTION
        frames.append((start, end, history, grid, labels))

    if not frames:
        raise RuntimeError("Could not create training frames from the canonical history")
    columns = []
    for _, _, frame, _, _ in frames:
        columns.extend(name for name in frame.columns if name not in columns)
    class_labels = [NO_ACTION, *action_classes, OTHER_ACTION]
    label_ids = {label: index for index, label in enumerate(class_labels)}
    input_sequences = []
    output_labels = []
    for start, end, frame, grid, labels in frames:
        values = frame.reindex(columns=columns).to_numpy(dtype=np.float64)
        first_index = SEQUENCE_LENGTH - 1
        last_index = np.searchsorted(grid, end - 900, side="right") - 1
        for index in range(first_index, last_index + 1):
            input_sequences.append(values[index - SEQUENCE_LENGTH + 1:index + 1])
            output_labels.append(label_ids[labels[index]])

    if not input_sequences:
        raise RuntimeError("Not enough samples for 60-minute sequence training")
    return (np.asarray(input_sequences, dtype=np.float32), np.asarray(output_labels, dtype=np.int64),
            columns, class_labels, last_end)


def train_model(database_path, artifact_path, timezone_name="UTC", epochs=24):
    database_path = Path(database_path).resolve()
    connection = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    try:
        sequences, labels, columns, class_labels, history_end = _history_frames(connection, timezone_name)
    finally:
        connection.close()

    train_values = sequences.reshape(-1, sequences.shape[-1]).astype(np.float64)
    observed = np.isfinite(train_values).mean(axis=0)
    usable = observed >= 0.05
    if not usable.any():
        raise RuntimeError("Training history contains no usable model features")
    sequences = sequences[:, :, usable]
    columns = [name for name, keep in zip(columns, usable) if keep]
    train_values = sequences.reshape(-1, sequences.shape[-1]).astype(np.float64)
    medians = np.nanmedian(train_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(train_values), train_values, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    sequences = np.where(np.isfinite(sequences), sequences, medians)
    sequences = np.clip((sequences - means) / scales, -8, 8).astype(np.float32)

    device = torch.device(preferred_device())
    torch.manual_seed(7)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = features.TemporalActionNetwork(
        sequences.shape[-1], hidden_size=24, num_classes=len(class_labels)).to(device)
    counts = np.bincount(labels, minlength=len(class_labels)).astype(np.float32)
    weights = np.sqrt(np.maximum(counts.max(), 1) / np.maximum(counts, 1))
    weights[0] = 1.0
    weights = np.clip(weights, 1.0, 8.0)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    training_data = torch.from_numpy(sequences).to(device)
    training_labels = torch.from_numpy(labels).to(device)
    batch_size = 256
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(labels), device=device)
        for batch in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(training_data[batch]), training_labels[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    artifact = ActionModel(model, class_labels, columns, medians, means, scales,
                           device, history_end, timezone_name)
    artifact.save(artifact_path)
    return artifact, {"samples": len(labels), "class_counts": dict(zip(class_labels, counts.astype(int))),
                      "features": len(columns), "sequence_minutes": SEQUENCE_LENGTH * 5,
                      "device": str(device), "history_end": history_end}