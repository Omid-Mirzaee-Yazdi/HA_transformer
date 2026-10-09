"""Build one normalized, deduplicated Home Assistant history database."""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class ProtectedBackupError(RuntimeError):
    pass


def readonly_connection(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def table_names(connection):
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def column_names(connection, table):
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def stable_text(value):
    return value.hex() if isinstance(value, bytes) else value


def fingerprint(values):
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def value_expr(alias, column, fallback="NULL"):
    return f"{alias}.{column}" if column else fallback


def context_expr(alias, name, columns):
    fields = [f"{alias}.{column}" for column in (f"{name}_bin", name) if column in columns]
    if not fields:
        return "NULL"
    return f"COALESCE({','.join(fields)})" if len(fields) > 1 else fields[0]


def state_query(connection):
    tables = table_names(connection)
    columns = column_names(connection, "states")
    if "metadata_id" in columns and "states_meta" in tables:
        entity = "m.entity_id"
        joins = " LEFT JOIN states_meta m ON m.metadata_id=s.metadata_id"
    elif "entity_id" in columns:
        entity = "s.entity_id"
        joins = ""
    else:
        raise RuntimeError("Could not resolve entity IDs in the states table")

    state_attrs = "NULL"
    if "attributes" in columns:
        state_attrs = "s.attributes"
    if "attributes_id" in columns and "state_attributes" in tables:
        joins += " LEFT JOIN state_attributes a ON a.attributes_id=s.attributes_id"
        state_attrs = "COALESCE(s.attributes,a.shared_attrs)" if state_attrs != "NULL" else "a.shared_attrs"

    updated = "last_updated_ts" if "last_updated_ts" in columns else "last_updated"
    changed = "last_changed_ts" if "last_changed_ts" in columns else "last_changed"
    return f"""SELECT {entity},s.state,s.{updated},s.{changed},
        {value_expr('s', 'last_reported_ts' if 'last_reported_ts' in columns else None)},
        {state_attrs},{context_expr('s', 'context_id', columns)},
        {context_expr('s', 'context_user_id', columns)},
        {context_expr('s', 'context_parent_id', columns)},{value_expr('s', 'origin_idx')}
        FROM states s{joins} WHERE s.state IS NOT NULL"""


def event_query(connection):
    tables = table_names(connection)
    columns = column_names(connection, "events")
    joins = ""
    event_type = value_expr("e", "event_type")
    if "event_type_id" in columns and "event_types" in tables:
        joins += " LEFT JOIN event_types t ON t.event_type_id=e.event_type_id"
        event_type = f"COALESCE({event_type},t.event_type)" if "event_type" in columns else "t.event_type"

    event_data = value_expr("e", "event_data")
    if "data_id" in columns and "event_data" in tables:
        joins += " LEFT JOIN event_data d ON d.data_id=e.data_id"
        event_data = f"COALESCE({event_data},d.shared_data)" if "event_data" in columns else "d.shared_data"

    fired = "time_fired_ts" if "time_fired_ts" in columns else "time_fired"
    return f"""SELECT {event_type},{event_data},{value_expr('e', 'origin')},
        {value_expr('e', 'origin_idx')},e.{fired},{context_expr('e', 'context_id', columns)},
        {context_expr('e', 'context_user_id', columns)},
        {context_expr('e', 'context_parent_id', columns)}
        FROM events e{joins}"""


def statistics_query(connection, table):
    tables = table_names(connection)
    columns = column_names(connection, table)
    joins = ""
    statistic_id = "m.statistic_id"
    if "metadata_id" not in columns or "statistics_meta" not in tables:
        raise RuntimeError(f"Could not resolve statistic IDs in {table}")
    joins = " JOIN statistics_meta m ON m.id=s.metadata_id"
    start = "start_ts" if "start_ts" in columns else "start"
    names = ("created_ts", "mean", "mean_weight", "min", "max", "last_reset_ts", "state", "sum")
    values = ",".join(value_expr("s", name) for name in names)
    return f"SELECT {statistic_id},s.{start},{values} FROM {table} s{joins}"


def database_span(path):
    connection = readonly_connection(path)
    columns = column_names(connection, "states")
    stamp = "last_updated_ts" if "last_updated_ts" in columns else "last_updated"
    span = connection.execute(f"SELECT MIN({stamp}),MAX({stamp}) FROM states").fetchone()
    connection.close()
    return tuple(timestamp(value) for value in span)


def extract_archive(archive_path, destination):
    destination.unlink(missing_ok=True)
    with tarfile.open(archive_path, "r:*") as outer:
        nested = next((member for member in outer if member.name.endswith("homeassistant.tar.gz")), None)
        if nested is None:
            raise RuntimeError("Missing homeassistant.tar.gz")
        stream = outer.extractfile(nested)
        if stream is None:
            raise RuntimeError("Could not read nested Home Assistant archive")
        if stream.read(9) == b"SecureTar":
            raise ProtectedBackupError(
                f"{Path(archive_path).name} is protected; decrypt it with the Home Assistant backup recovery key first"
            )
        stream = outer.extractfile(nested)
        with tarfile.open(fileobj=stream, mode="r|gz") as inner:
            for member in inner:
                if member.name.endswith("home-assistant_v2.db"):
                    content = inner.extractfile(member)
                    if content is None:
                        break
                    with destination.open("wb") as output:
                        shutil.copyfileobj(content, output)
                    return
    raise RuntimeError("Missing home-assistant_v2.db")


def create_schema(connection):
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE sources(source TEXT PRIMARY KEY, bytes INTEGER NOT NULL,
            min_state_ts REAL, max_state_ts REAL, state_rows_seen INTEGER NOT NULL,
            event_rows_seen INTEGER NOT NULL, long_term_rows_seen INTEGER NOT NULL,
            short_term_rows_seen INTEGER NOT NULL);
        CREATE TABLE states(record_hash TEXT PRIMARY KEY, entity_id TEXT NOT NULL,
            state TEXT NOT NULL, last_updated_ts REAL NOT NULL, last_changed_ts REAL,
            last_reported_ts REAL, attributes TEXT, context_id TEXT,
            context_user_id TEXT, context_parent_id TEXT, origin_idx INTEGER);
        CREATE INDEX ix_states_entity_updated ON states(entity_id,last_updated_ts);
        CREATE TABLE events(record_hash TEXT PRIMARY KEY, event_type TEXT,
            event_data TEXT, origin TEXT, origin_idx INTEGER, time_fired_ts REAL,
            context_id TEXT, context_user_id TEXT, context_parent_id TEXT);
        CREATE INDEX ix_events_time ON events(time_fired_ts);
        CREATE TABLE statistics_meta(statistic_id TEXT PRIMARY KEY, source TEXT,
            unit_of_measurement TEXT, has_mean INTEGER, has_sum INTEGER, name TEXT,
            mean_type INTEGER, unit_class TEXT);
        CREATE TABLE statistics(statistic_id TEXT NOT NULL, start_ts REAL NOT NULL,
            created_ts REAL, mean REAL, mean_weight REAL, min REAL, max REAL,
            last_reset_ts REAL, state REAL, sum REAL,
            PRIMARY KEY(statistic_id,start_ts));
        CREATE INDEX ix_statistics_start ON statistics(start_ts);
        CREATE TABLE statistics_short_term(statistic_id TEXT NOT NULL, start_ts REAL NOT NULL,
            created_ts REAL, mean REAL, mean_weight REAL, min REAL, max REAL,
            last_reset_ts REAL, state REAL, sum REAL,
            PRIMARY KEY(statistic_id,start_ts));
        CREATE INDEX ix_statistics_short_start ON statistics_short_term(start_ts);
    """)


def import_database(source_path, source_name, source_bytes, destination):
    source = readonly_connection(source_path)
    tables = table_names(source)
    counts = {"states": 0, "events": 0, "statistics": 0, "statistics_short_term": 0}

    state_rows = []
    for row in source.execute(state_query(source)):
        entity, state, updated, changed, reported, attrs, context, user, parent, origin_idx = row
        if entity is None or updated is None:
            continue
        values = (str(entity), str(state), timestamp(updated), timestamp(changed),
                  timestamp(reported), stable_text(attrs), stable_text(context),
                  stable_text(user), stable_text(parent), origin_idx)
        state_rows.append((fingerprint(values), *values))
        counts["states"] += 1
        if len(state_rows) >= 5000:
            destination.executemany("INSERT OR IGNORE INTO states VALUES(?,?,?,?,?,?,?,?,?,?,?)", state_rows)
            state_rows.clear()
    if state_rows:
        destination.executemany("INSERT OR IGNORE INTO states VALUES(?,?,?,?,?,?,?,?,?,?,?)", state_rows)

    if "events" in tables:
        event_rows = []
        for row in source.execute(event_query(source)):
            event_type, data, origin, origin_idx, fired, context, user, parent = row
            if fired is None:
                continue
            values = (stable_text(event_type), stable_text(data), stable_text(origin),
                      origin_idx, timestamp(fired), stable_text(context),
                      stable_text(user), stable_text(parent))
            event_rows.append((fingerprint(values), *values))
            counts["events"] += 1
            if len(event_rows) >= 5000:
                destination.executemany("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?)", event_rows)
                event_rows.clear()
        if event_rows:
            destination.executemany("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?)", event_rows)

    if "statistics_meta" in tables:
        meta_columns = column_names(source, "statistics_meta")
        meta_fields = ("source", "unit_of_measurement", "has_mean", "has_sum", "name", "mean_type", "unit_class")
        selected = ["statistic_id"] + [field if field in meta_columns else f"NULL AS {field}" for field in meta_fields]
        meta_rows = source.execute(f"SELECT {','.join(selected)} FROM statistics_meta")
        destination.executemany(
            "INSERT INTO statistics_meta VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(statistic_id) DO UPDATE SET "
            "source=excluded.source,unit_of_measurement=excluded.unit_of_measurement,"
            "has_mean=excluded.has_mean,has_sum=excluded.has_sum,name=excluded.name,"
            "mean_type=excluded.mean_type,unit_class=excluded.unit_class", meta_rows)

        for table in ("statistics", "statistics_short_term"):
            if table not in tables:
                continue
            query = statistics_query(source, table)
            for row in source.execute(query):
                statistic_id, start, *values = row
                if statistic_id is None or start is None:
                    continue
                normalized = (statistic_id, timestamp(start), *(timestamp(value) if index in (0, 5) else value
                              for index, value in enumerate(values)))
                destination.execute(
                    f"INSERT INTO {table} VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(statistic_id,start_ts) DO UPDATE SET "
                    "created_ts=excluded.created_ts,mean=excluded.mean,mean_weight=excluded.mean_weight,"
                    "min=excluded.min,max=excluded.max,last_reset_ts=excluded.last_reset_ts,"
                    "state=excluded.state,sum=excluded.sum", normalized)
                counts[table] += 1

    min_ts, max_ts = database_span(source_path)
    destination.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)",
                        (source_name, source_bytes, min_ts, max_ts, counts["states"],
                         counts["events"], counts["statistics"], counts["statistics_short_term"]))
    source.close()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-database", type=Path, default=Path("home-assistant_v2.db"))
    parser.add_argument("--backup-dir", type=Path, default=Path("data/source_backups"))
    parser.add_argument("--output", type=Path, default=Path("data/ha_history.sqlite3"))
    args = parser.parse_args()
    live_database = args.live_database.resolve()
    if not live_database.is_file():
        raise SystemExit(f"Live database not found: {live_database}")
    archives = sorted(args.backup_dir.resolve().glob("*.tar"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ha-merge-") as temporary:
        temporary = Path(temporary)
        inputs = [(live_database, "live", live_database.stat().st_size, database_span(live_database))]
        for index, archive in enumerate(archives):
            extracted = temporary / f"inspect-{index}.db"
            try:
                extract_archive(archive, extracted)
                inputs.append((archive, archive.name, archive.stat().st_size, database_span(extracted)))
            except ProtectedBackupError as error:
                raise SystemExit(str(error)) from error
            except (OSError, sqlite3.Error, tarfile.TarError, RuntimeError) as error:
                print(f"Skipped {archive.name}: {error}")
            finally:
                extracted.unlink(missing_ok=True)
        inputs.sort(key=lambda item: (item[3][1] or float("-inf"), item[1]))

        temporary_output = temporary / "ha_history.sqlite3"
        destination = sqlite3.connect(temporary_output)
        create_schema(destination)
        for source_path, source_name, source_bytes, _ in inputs:
            extracted = temporary / "source.db"
            actual_path = source_path
            if source_path.suffix == ".tar":
                extract_archive(source_path, extracted)
                actual_path = extracted
            try:
                counts = import_database(actual_path, source_name, source_bytes, destination)
                destination.commit()
                print(f"{source_name}: states={counts['states']:,} events={counts['events']:,} "
                      f"statistics={counts['statistics']:,} short_term={counts['statistics_short_term']:,}")
            finally:
                if actual_path == extracted:
                    extracted.unlink(missing_ok=True)
        destination.execute("ANALYZE")
        destination.commit()
        destination.close()
        os.replace(temporary_output, args.output.resolve())

    check = sqlite3.connect(args.output.resolve().as_uri() + "?mode=ro", uri=True)
    for table in ("sources", "states", "events", "statistics_meta", "statistics", "statistics_short_term"):
        print(f"Merged {table}: {check.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:,} rows")
    check.close()


if __name__ == "__main__":
    main()