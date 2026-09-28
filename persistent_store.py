"""Small durable state store shared by the web app and scheduled jobs.

When ``DATABASE_URL`` is configured, mutable files are stored in PostgreSQL.
Local development keeps using the existing files.  PostgreSQL is deliberately
preferred over Render's ephemeral filesystem; a configured database failure is
reported instead of silently writing data that will disappear on restart.
"""

from __future__ import annotations

import os
from pathlib import Path


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
NAMESPACE = os.getenv("PROJECT_X_ENV", "production").strip() or "production"


def _connect():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - deployment configuration error
        raise RuntimeError("DATABASE_URL is set, but psycopg is not installed.") from exc
    return psycopg.connect(DATABASE_URL, connect_timeout=10)


def _ensure_table(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS project_x_state (
            namespace TEXT NOT NULL,
            state_key TEXT NOT NULL,
            payload BYTEA NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (namespace, state_key)
        )
        """
    )


def using_database() -> bool:
    return bool(DATABASE_URL)


def read_bytes(path: Path) -> bytes | None:
    """Read a mutable asset from PostgreSQL, or from its local file fallback."""
    if not DATABASE_URL:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None
    with _connect() as connection:
        _ensure_table(connection)
        row = connection.execute(
            "SELECT payload FROM project_x_state WHERE namespace=%s AND state_key=%s",
            (NAMESPACE, path.name),
        ).fetchone()
    return bytes(row[0]) if row else None


def write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace one mutable asset in the selected durable store."""
    if not DATABASE_URL:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return
    with _connect() as connection:
        _ensure_table(connection)
        connection.execute(
            """
            INSERT INTO project_x_state(namespace, state_key, payload, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT(namespace, state_key) DO UPDATE
            SET payload=EXCLUDED.payload, updated_at=NOW()
            """,
            (NAMESPACE, path.name, payload),
        )


def list_database_keys() -> list[tuple[str, str]]:
    """Return state keys and update timestamps for diagnostics/backups."""
    if not DATABASE_URL:
        return []
    with _connect() as connection:
        _ensure_table(connection)
        rows = connection.execute(
            "SELECT state_key, updated_at::text FROM project_x_state WHERE namespace=%s ORDER BY state_key",
            (NAMESPACE,),
        ).fetchall()
    return [(str(key), str(updated)) for key, updated in rows]
