"""Backup and migrate Project X mutable state without exposing it to Git."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from persistent_store import list_database_keys, read_bytes, using_database, write_bytes


ROOT = Path(__file__).resolve().parent
MUTABLE = (
    "project_x_portfolio.csv",
    "project_x_trade_journal.csv",
    "project_x_settings.json",
    "prediction_history.csv",
    "project_x_ranking_history.csv",
    "next_day_prediction_history.csv",
    "next_day_prediction_results.csv",
    "candidate_observations.sqlite",
    "watchlist_fresh_candidates.csv",
    "watchlist_update_status.json",
    "verified_materials.json",
    "daytrade_JP.json",
    "daytrade_US.json",
    "cloud_job_status.json",
)


def backup(local_only: bool = False) -> Path:
    folder = ROOT / ".local_backups"
    folder.mkdir(exist_ok=True)
    target = folder / f"project_x_state_backup_{datetime.now():%Y%m%d_%H%M%S}.zip"
    manifest = {"created_at": datetime.now().isoformat(), "database": using_database(), "files": []}
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name in MUTABLE:
            path = ROOT / name
            payload = path.read_bytes() if local_only and path.exists() else read_bytes(path)
            if payload is not None:
                archive.writestr(name, payload)
                manifest["files"].append({"name": name, "bytes": len(payload)})
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return target


def migrate_local() -> int:
    if not using_database():
        raise RuntimeError("DATABASE_URLを設定してから実行してください。")
    count = 0
    for name in MUTABLE:
        path = ROOT / name
        if path.exists():
            write_bytes(path, path.read_bytes())
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("backup", "migrate", "status"))
    args = parser.parse_args()
    if args.command == "backup":
        print(backup())
    elif args.command == "migrate":
        destination = backup(local_only=True)
        print(f"backup={destination}")
        print(f"migrated={migrate_local()}")
    else:
        print(json.dumps({"database": using_database(), "keys": list_database_keys()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
