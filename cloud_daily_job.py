"""Cloud scheduler entry point; never starts Streamlit or trains models."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from persistent_store import write_bytes


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "cloud_job_status.json"


def save_status(state: str, step: str, **extra) -> None:
    payload = {
        "state": state,
        "step": step,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        **extra,
    }
    write_bytes(STATUS, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def main() -> None:
    started = pd.Timestamp.now(tz="UTC").isoformat()
    try:
        save_status("RUNNING", "jp_watchlist", started_at=started)
        from refresh_watchlist import main as refresh_watchlist
        refresh_watchlist()

        save_status("RUNNING", "jp_daytrade", started_at=started)
        from daytrade_mode import scan
        scan("JP", ROOT)

        save_status("RUNNING", "us_daytrade", started_at=started)
        scan("US", ROOT)

        save_status("COMPLETED", "all", started_at=started)
    except Exception as exc:
        save_status(
            "FAILED",
            "unknown",
            started_at=started,
            error=str(exc),
            traceback=traceback.format_exc(limit=8),
        )
        raise


if __name__ == "__main__":
    main()
