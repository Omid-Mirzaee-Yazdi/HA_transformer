# Home Assistant history

`ha_history.sqlite3` is the normalized, deduplicated history database used by
`world_model.py`. It is generated locally and intentionally excluded from Git.

The original backup archives are preserved in `source_backups/`. They are not
exact duplicates: each contributes unique recorder history. The live recorder
database remains at the project root and is read without modification.

Rebuild the canonical database after adding or replacing source snapshots:

```sh
python3 merge_recorder.py
```

Run the offline world-model evaluation against the canonical database, using the
Home Assistant IANA timezone:

```sh
python3 world_model.py --timezone Europe/London
```

The model evaluates occupancy persistence baselines, clock/context action
propensity, and a GRU over the preceding 60 minutes of environmental and
user-targeted device state. User-linked calls are grouped by Home Assistant
context and exact-action results are reported only for repeated classes. Each
run trains and evaluates chronologically; it does not save a checkpoint or call
Home Assistant services. Current action labels are sparse, so backtest metrics
are exploratory and are not sufficient to authorize unattended device control.

The canonical database stores semantic state and event records, resolved
statistics metadata, and long- and short-term statistics. State and event
duplicates are removed by content fingerprints. Statistics are keyed by
statistic ID and timestamp; later snapshots take precedence if a row differs.
The `sources` table records input names, coverage, and row counts.