"""Read-only Home Assistant occupancy and user-action forecasting experiments."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONTEXT_TERMS = (
    "illuminance", "light_level", "lux", "temperature", "humidity", "pressure",
    "cloud", "wind", "outdoor", "co2", "voc", "pm25", "motion", "presence",
    "occupancy", "person.", "device_tracker.",
)
WEATHER_ATTRIBUTES = (
    "temperature", "humidity", "cloud_coverage", "pressure", "wind_speed",
    "wind_gust_speed", "visibility", "uv_index", "precipitation",
)
TRUE_STATES = {"on", "home", "occupied", "detected", "open", "true", "yes"}
FALSE_STATES = {"off", "not_home", "not_occupied", "clear", "closed", "false", "no"}
ACTION_DOMAINS = {
    "alarm_control_panel", "button", "climate", "cover", "fan", "input_boolean",
    "input_select", "light", "lock", "media_player", "scene", "script", "select",
    "switch", "vacuum",
}
ACTION_LOOKBACKS = ((3600, "1h"), (86400, "24h"), (604800, "7d"))


class TemporalActionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=48, num_classes=1):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.temporal_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_size, num_classes))

    def forward(self, values):
        projected = self.input_projection(values)
        encoded, _ = self.temporal_encoder(projected)
        logits = self.output(encoded[:, -1])
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


def get_entities(connection):
    columns = {row[1] for row in connection.execute("PRAGMA table_info(states)")}
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "metadata_id" in columns and "states_meta" in tables:
        entity = "m.entity_id"
        join = " JOIN states_meta m ON m.metadata_id=s.metadata_id"
    elif "entity_id" in columns:
        entity = "s.entity_id"
        join = ""
    else:
        raise RuntimeError("Could not locate entity IDs in the states table")
    matches = " OR ".join(f"lower({entity}) LIKE '%{term}%'" for term in CONTEXT_TERMS)
    return entity, join, columns, matches


def get_control_entities(connection, start, end):
    entities = set()
    for (payload,) in connection.execute(
            """SELECT event_data FROM events WHERE event_type='call_service'
            AND context_user_id IS NOT NULL AND time_fired_ts BETWEEN ? AND ?""",
            (start, end)):
        try:
            event = json.loads(payload or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if event.get("domain") not in ACTION_DOMAINS:
            continue
        service_data = event.get("service_data") or {}
        target = (event.get("target") or {}).get("entity_id") or service_data.get("entity_id")
        if isinstance(target, dict):
            target = target.get("entity_id")
        if isinstance(target, str):
            entities.add(target)
        elif isinstance(target, list):
            entities.update(item for item in target if isinstance(item, str))
    available = {row[0] for row in connection.execute("SELECT DISTINCT entity_id FROM states")}
    return sorted(entities & available)


def timestamp_seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    return pd.Timestamp(value).timestamp()


def to_binary(value):
    value = str(value).strip().lower()
    if value in TRUE_STATES:
        return 1.0
    if value in FALSE_STATES:
        return 0.0
    return np.nan


def align_as_of(series, grid):
    index = series.index.union(grid).sort_values()
    return series.reindex(index).ffill().reindex(grid)


def history_windows(connection):
    times = [r[0] for r in connection.execute(
        "SELECT DISTINCT last_updated_ts FROM states ORDER BY last_updated_ts")]
    windows = []
    if not times:
        return windows
    begin = previous = times[0]
    for value in times[1:]:
        if value - previous > 7 * 86400:
            windows.append((begin, previous))
            begin = value
        previous = value
    windows.append((begin, previous))
    return [(begin, end) for begin, end in windows if end - begin >= 86400]


def aligned_series(connection, entity, start, end, grid):
    rows = connection.execute(
        "SELECT last_updated_ts,state FROM states WHERE entity_id=? AND last_updated_ts BETWEEN ? AND ? ORDER BY last_updated_ts",
        (entity, start - 3600, end),
    ).fetchall()
    if not rows:
        return None
    raw = [r[1] for r in rows]
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce")
    if numeric.notna().sum() >= max(1, len(raw) // 2):
        values = numeric.tolist()
    elif entity.startswith(("binary_sensor.", "person.", "device_tracker.", "automation.",
                            "cover.", "fan.", "input_boolean.", "light.", "lock.",
                            "script.", "switch.")) or entity == "sun.sun":
        values = [to_binary(value) for value in raw]
    else:
        values = raw
    index = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
    series = pd.Series(values, index=index).groupby(level=0).last().sort_index()
    return align_as_of(series, grid)


def make_features(connection, entities, targets, start, end, timezone, require_targets=True):
    grid = pd.date_range(pd.Timestamp(start, unit="s", tz="UTC").floor("5min"),
                         pd.Timestamp(end, unit="s", tz="UTC").floor("5min"), freq="5min")
    columns = {}
    current_targets = {}
    for entity in entities:
        series = aligned_series(connection, entity, start, end, grid)
        if series is None:
            continue
        if entity in targets:
            current_targets[entity] = pd.to_numeric(series, errors="coerce").where(series.isin([0, 1]))
        if pd.api.types.is_numeric_dtype(series):
            columns[entity] = pd.to_numeric(series, errors="coerce")
        elif entity.startswith("weather.") or entity == "sun.sun":
            encoded = pd.get_dummies(series.astype("string"), prefix=entity, dtype=float)
            columns.update({name: encoded[name] for name in encoded})

        if entity.startswith("weather."):
            for updated, attrs in connection.execute(
                "SELECT last_updated_ts,attributes FROM states WHERE entity_id=? AND last_updated_ts BETWEEN ? AND ? ORDER BY last_updated_ts",
                (entity, start - 3600, end),
            ):
                try:
                    attrs = json.loads(attrs or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                for name in WEATHER_ATTRIBUTES:
                    value = attrs.get(name)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        columns.setdefault(f"{entity}.{name}", []).append((updated, float(value)))

    if require_targets and not current_targets:
        return None
    for name, values in list(columns.items()):
        if isinstance(values, list):
            index = pd.to_datetime([item[0] for item in values], unit="s", utc=True)
            series = pd.Series([item[1] for item in values], index=index).groupby(level=0).last()
            columns[name] = align_as_of(series, grid)
    frame = pd.DataFrame(columns, index=grid)
    local = grid.tz_convert(timezone)
    minutes = local.hour * 60 + local.minute
    weekday_minutes = local.dayofweek * 1440 + minutes
    frame["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    frame["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    frame["week_sin"] = np.sin(2 * np.pi * weekday_minutes / 10080)
    frame["week_cos"] = np.cos(2 * np.pi * weekday_minutes / 10080)
    frame["weekend"] = (local.dayofweek >= 5).astype(float)
    features = pd.concat([frame.shift(lag).add_suffix(f"_lag{lag}") for lag in (0, 1, 3, 6)], axis=1)
    future = {name: series.shift(-3) for name, series in current_targets.items()}
    return features.replace([np.inf, -np.inf], np.nan), current_targets, future


def score_window(features, current, future):
    split = int(len(features) * 0.75)
    if split < 300 or len(features) - split < 100:
        return []
    results = []
    for entity, target in future.items():
        labels = target.to_numpy()
        train = (np.arange(len(labels)) < split) & np.isin(labels, [0, 1])
        test = (np.arange(len(labels)) >= split) & np.isin(labels, [0, 1])
        y_train, y_test = labels[train].astype(int), labels[test].astype(int)
        if len(y_train) < 200 or len(y_test) < 80 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        x_train, x_test = features.iloc[np.flatnonzero(train)], features.iloc[np.flatnonzero(test)]
        usable = x_train.columns[x_train.notna().any()]
        if not len(usable):
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, l2_regularization=2, random_state=7),
        )
        model.fit(x_train[usable], y_train)
        prediction = model.predict(x_test[usable])
        persistence = current[entity].to_numpy()[np.flatnonzero(test)].astype(int)
        results.append((entity, len(y_train), len(y_test),
                        balanced_accuracy_score(y_test, prediction),
                        balanced_accuracy_score(y_test, persistence)))
    return results


def direct_user_actions(connection, start, end):
    return np.asarray([time for time, _ in direct_user_action_records(connection, start, end)], dtype=float)


def direct_user_action_records(connection, start, end):
    rows = connection.execute(
        """SELECT time_fired_ts,context_id,event_data FROM events
        WHERE event_type='call_service' AND context_user_id IS NOT NULL
        AND time_fired_ts BETWEEN ? AND ? ORDER BY time_fired_ts""",
        (start, end),
    )
    episodes = {}
    for fired, context_id, payload in rows:
        try:
            event = json.loads(payload or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if event.get("domain") not in ACTION_DOMAINS:
            continue
        service_data = event.get("service_data") or {}
        target = (event.get("target") or {}).get("entity_id") or service_data.get("entity_id")
        if isinstance(target, dict):
            target = target.get("entity_id")
        if isinstance(target, str):
            target = [target]
        if not isinstance(target, list) or not target:
            continue
        entities = sorted(str(item) for item in target if isinstance(item, str))
        if not entities:
            continue
        label = f"{event['domain']}.{event['service']} -> {','.join(entities)}"
        key = context_id or f"{fired}:{payload}"
        current = episodes.get(key)
        if current is None or fired < current[0]:
            episodes[key] = (float(fired), label)
    return sorted(episodes.values())


def action_history_features(features, action_times):
    result = features.copy()
    seconds = result.index.asi8 / 1e9
    end = np.searchsorted(action_times, seconds, side="left")
    for lookback, label in ACTION_LOOKBACKS:
        begin = np.searchsorted(action_times, seconds - lookback, side="left")
        result[f"user_actions_{label}"] = end - begin
    previous = np.maximum(end - 1, 0)
    elapsed = seconds - action_times[previous] if len(action_times) else np.full(len(seconds), np.nan)
    result["minutes_since_user_action"] = np.where(end > 0, elapsed / 60, np.nan)
    return result


def action_model_scores(train_features, test_features, y_train, y_test):
    x_train = train_features.replace([np.inf, -np.inf], np.nan)
    x_test = test_features.replace([np.inf, -np.inf], np.nan)
    usable = x_train.columns[x_train.notna().any()]
    if not len(usable):
        return {"average_precision": float(y_test.mean()), "roc_auc": 0.5,
                "brier": float(y_test.mean() * (1 - y_test.mean())),
                "top10_precision": float(y_test.mean()), "top10_lift": 1.0}
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=2000, random_state=7),
    )
    model.fit(x_train[usable], y_train)
    probability = model.predict_proba(x_test[usable])[:, 1]
    top_count = max(1, int(np.ceil(len(y_test) * 0.10)))
    top_indices = np.argsort(probability)[-top_count:]
    return {
        "average_precision": average_precision_score(y_test, probability),
        "roc_auc": roc_auc_score(y_test, probability),
        "brier": brier_score_loss(y_test, probability),
        "top10_precision": float(y_test[top_indices].mean()),
        "top10_lift": float(y_test[top_indices].mean() / y_test.mean()),
    }


def sequence_action_scores(features, labels, train, test, sequence_length=12):
    sequence_columns = [column for column in features.columns
                        if column.endswith("_lag0") or column.startswith("user_actions_")
                        or column == "minutes_since_user_action"]
    features = features[sequence_columns]
    train_indices = np.flatnonzero(train)
    test_indices = np.flatnonzero(test)
    train_indices = train_indices[train_indices >= sequence_length - 1]
    test_indices = test_indices[test_indices >= sequence_length - 1]
    y_train = labels[train_indices].astype(np.float32)
    y_test = labels[test_indices].astype(int)
    if y_train.sum() < 20 or y_test.sum() < 8 or len(np.unique(y_test)) < 2:
        return None

    values = features.to_numpy(dtype=np.float64)
    training_end = train_indices[-1] + 1
    training_values = values[:training_end]
    observed = np.isfinite(training_values).mean(axis=0)
    usable = observed >= 0.05
    if not usable.any():
        return None
    training_values = training_values[:, usable]
    medians = np.nanmedian(training_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    means = np.nanmean(np.where(np.isfinite(training_values), training_values, medians), axis=0)
    standard_deviations = np.nanstd(
        np.where(np.isfinite(training_values), training_values, medians), axis=0)
    standard_deviations = np.where(standard_deviations > 1e-6, standard_deviations, 1.0)
    normalized = np.where(np.isfinite(values[:, usable]), values[:, usable], medians)
    normalized = np.clip((normalized - means) / standard_deviations, -8, 8).astype(np.float32)

    def sequences(indices):
        return np.stack([normalized[index - sequence_length + 1:index + 1] for index in indices])

    torch.manual_seed(7)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = TemporalActionNetwork(normalized.shape[1], hidden_size=24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(min(20.0, negatives / positives), dtype=torch.float32))
    train_values = torch.from_numpy(sequences(train_indices))
    train_labels = torch.from_numpy(y_train)
    model.train()
    for _ in range(10):
        order = torch.randperm(len(train_indices))
        for batch in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_values[batch])
            loss = loss_function(logits, train_labels[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(sequences(test_indices)))).numpy()
    top_count = max(1, int(np.ceil(len(y_test) * 0.10)))
    top_indices = np.argsort(probabilities)[-top_count:]
    base_rate = float(y_test.mean())
    return {
        "average_precision": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "brier": brier_score_loss(y_test, probabilities),
        "top10_precision": float(y_test[top_indices].mean()),
        "top10_lift": float(y_test[top_indices].mean() / base_rate),
        "sequence_minutes": sequence_length * 5,
        "features": int(usable.sum()),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
    }


def score_user_action_window(features, action_times, window_end):
    seconds = features.index.asi8 / 1e9
    next_action = np.searchsorted(action_times, seconds, side="right")
    labels = np.zeros(len(seconds), dtype=int)
    available = next_action < len(action_times)
    labels[available] = action_times[next_action[available]] <= seconds[available] + 900
    complete = seconds + 900 <= window_end
    split = int(len(features) * 0.75)
    train = (np.arange(len(features)) < split - 3) & complete
    test = (np.arange(len(features)) >= split) & complete
    y_train, y_test = labels[train], labels[test]
    positives_train, positives_test = int(y_train.sum()), int(y_test.sum())
    if positives_train < 20 or positives_test < 8 or len(np.unique(y_test)) < 2:
        return {"skipped": True, "train_positive": positives_train,
                "test_positive": positives_test, "test_size": int(test.sum())}

    clock_columns = [column for column in features.columns
                     if column.startswith(("tod_", "week_", "weekend_"))]
    if not clock_columns:
        return {"skipped": True, "train_positive": positives_train,
                "test_positive": positives_test, "test_size": int(test.sum())}
    clock = features[clock_columns]
    context = action_history_features(features, action_times)

    return {
        "skipped": False,
        "train_positive": positives_train,
        "test_positive": positives_test,
        "test_size": int(test.sum()),
        "base_rate": float(y_test.mean()),
        "clock": action_model_scores(clock.loc[train], clock.loc[test], y_train, y_test),
        "context": action_model_scores(context.loc[train], context.loc[test], y_train, y_test),
        "sequence": sequence_action_scores(context, labels, train, test),
    }


def score_exact_action_window(features, action_records, window_end):
    seconds = features.index.asi8 / 1e9
    split = int(len(features) * 0.75)
    train = (np.arange(len(features)) < split - 3) & (seconds + 900 <= window_end)
    test = (np.arange(len(features)) >= split) & (seconds + 900 <= window_end)
    split_time = seconds[split]
    counts = {}
    for fired, label in action_records:
        if fired < split_time - 900:
            counts[label] = counts.get(label, 0) + 1
    candidates = [label for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
                  if count >= 8][:5]
    if not candidates or not test.any():
        return []

    clock_columns = [column for column in features.columns
                     if column.startswith(("tod_", "week_", "weekend_"))]
    clock = features[clock_columns]
    context = action_history_features(features, np.asarray(sorted({time for time, _ in action_records})))
    results = []
    for label in candidates:
        label_times = np.asarray(sorted(fired for fired, action in action_records if action == label))
        test_episodes = int(sum(fired >= split_time for fired, action in action_records if action == label))
        if test_episodes < 4:
            continue
        next_action = np.searchsorted(label_times, seconds, side="right")
        labels = np.zeros(len(seconds), dtype=int)
        available = next_action < len(label_times)
        labels[available] = label_times[next_action[available]] <= seconds[available] + 900
        y_train, y_test = labels[train], labels[test]
        if y_train.sum() < 20 or y_test.sum() < 8 or len(np.unique(y_test)) < 2:
            continue
        results.append({
            "label": label,
            "train_episodes": counts[label],
            "test_episodes": test_episodes,
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
            "test_size": int(test.sum()),
            "base_rate": float(y_test.mean()),
            "clock": action_model_scores(clock.loc[train], clock.loc[test], y_train, y_test),
            "context": action_model_scores(context.loc[train], context.loc[test], y_train, y_test),
            "sequence": sequence_action_scores(context, labels, train, test),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(Path(__file__).resolve().parent / "data" / "ha_history.sqlite3"))
    parser.add_argument("--timezone", default="UTC", help="IANA timezone used by Home Assistant")
    args = parser.parse_args()
    database = Path(args.database).resolve()
    if not database.exists():
        raise SystemExit(f"Canonical database not found: {database}. Run merge_recorder.py first.")
    staged = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    entity, join, _, matches = get_entities(staged)
    context_entities = [r[0] for r in staged.execute(
        f"SELECT DISTINCT {entity} FROM states s{join} WHERE ({matches}) ORDER BY {entity}")]
    targets = {e for e in context_entities if e.startswith("binary_sensor.") and any(k in e.lower() for k in ("presence", "occupancy"))}
    windows = history_windows(staged)
    if not targets or not windows:
        raise SystemExit("No usable occupancy targets or history windows found.")
    print(f"Source: {database}")
    print(f"Environmental entities: {len(context_entities)}; "
          f"occupancy targets: {len(targets)}; actuator features use training-period user actions")
    print(f"Windows: {len(windows)}; grid: 5 minutes; prediction horizon: 15 minutes; timezone: {args.timezone}")
    all_results = []
    action_results = []
    for start, end in windows:
        label = f"{pd.Timestamp(start, unit='s', tz='UTC').date()}..{pd.Timestamp(end, unit='s', tz='UTC').date()}"
        grid = pd.date_range(pd.Timestamp(start, unit="s", tz="UTC").floor("5min"),
                             pd.Timestamp(end, unit="s", tz="UTC").floor("5min"), freq="5min")
        split_time = grid[int(len(grid) * 0.75)].timestamp()
        control_entities = get_control_entities(staged, start, split_time - 1)
        entities = sorted(set(context_entities) | set(control_entities))
        data = make_features(staged, entities, targets, start, end, args.timezone)
        if data:
            features, current, future = data
            results = score_window(features, current, future)
            all_results.extend((label, *result) for result in results)
            print(f"{label}: {len(features):,} five-minute states; "
                  f"training-known controls={len(control_entities)}; scored {len(results)} targets")
            action_records = direct_user_action_records(staged, start, end)
            action_times = np.asarray([fired for fired, _ in action_records], dtype=float)
            action_results.append((label,
                                   score_user_action_window(features, action_times, end),
                                   score_exact_action_window(features, action_records, end)))
    if not all_results:
        raise SystemExit("Not enough two-class occupancy data for chronological evaluation.")
    print("\nHeld-out balanced accuracy (higher is better):")
    for label, entity, n_train, n_test, model_score, baseline_score in all_results:
        print(f"{entity} | {label} | train={n_train:,} test={n_test:,} | context={model_score:.3f} persistence={baseline_score:.3f} delta={model_score-baseline_score:+.3f}")
    print("\nDirect user-action propensity for the next 15 minutes:")
    for label, result, exact_results in action_results:
        if result["skipped"]:
            print(f"{label} | skipped: train positives={result['train_positive']}, "
                  f"test positives={result['test_positive']}")
            continue
        print(f"{label} | train positives={result['train_positive']} "
              f"test positives={result['test_positive']}/{result['test_size']} "
              f"({result['base_rate']:.1%} base rate)")
        for name in ("clock", "context"):
            scores = result[name]
            print(f"  {name}: AP={scores['average_precision']:.3f} "
                  f"AUC={scores['roc_auc']:.3f} "
                  f"top10={scores['top10_precision']:.1%} "
                  f"lift={scores['top10_lift']:.2f}x "
                  f"Brier={scores['brier']:.4f}")
        sequence_scores = result["sequence"]
        if sequence_scores:
            print(f"  sequence ({sequence_scores['sequence_minutes']}m, "
                  f"{sequence_scores['features']} features): "
                  f"AP={sequence_scores['average_precision']:.3f} "
                  f"AUC={sequence_scores['roc_auc']:.3f} "
                  f"top10={sequence_scores['top10_precision']:.1%} "
                  f"lift={sequence_scores['top10_lift']:.2f}x "
                  f"Brier={sequence_scores['brier']:.4f}")
        else:
            print("  sequence: skipped; insufficient positive examples")
        print(f"  constant-rate Brier={result['base_rate'] * (1 - result['base_rate']):.4f}")
        if exact_results:
            print("  Repeated exact actions:")
            for exact in exact_results:
                print(f"    {exact['label']} | episodes={exact['train_episodes']} train/"
                      f"{exact['test_episodes']} test | windows={exact['test_positive']}/"
                      f"{exact['test_size']}")
                for name in ("clock", "context"):
                    scores = exact[name]
                    print(f"      {name}: AP={scores['average_precision']:.3f} "
                          f"top10={scores['top10_precision']:.1%} "
                          f"lift={scores['top10_lift']:.2f}x")
                if exact["sequence"]:
                    scores = exact["sequence"]
                    print(f"      sequence: AP={scores['average_precision']:.3f} "
                          f"top10={scores['top10_precision']:.1%} "
                          f"lift={scores['top10_lift']:.2f}x "
                          f"Brier={scores['brier']:.4f}")
                else:
                    print("      sequence: skipped; insufficient positive examples")
        else:
            print("  No repeated exact action class has enough examples on both sides of the split.")
        print("\nAction scores rank five-minute forecast windows; exact-action results are one-vs-rest "
            "for repeated classes. This is an offline model evaluation; no Home Assistant services are called.")
    staged.close()


if __name__ == "__main__":
    main()