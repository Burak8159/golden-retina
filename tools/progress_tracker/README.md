# Golden Retina progress tracker

A small, read-only dashboard for the 38-block technical development plan.
It binds only to `127.0.0.1` (default port `8768`; this run used `8769`) and
reads aggregate progress from `results/technical_dev_20261108_rerun/status.json`.

Start it from the repository root:

```sh
python3 tools/progress_tracker/server.py
```

Open the URL printed by the server. The page refreshes status every 10 seconds.
To stop the server, press Ctrl-C in the terminal where it was started.

The queued post-program mBRSET execution sequence and data-handling rules are
documented in [`docs/mbrset_followup_execution.md`](../../docs/mbrset_followup_execution.md)
and displayed at the bottom of the tracker.
