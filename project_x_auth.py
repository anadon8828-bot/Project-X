"""Local password verification for the externally exposed Project X app."""

import hashlib
import hmac
import json
import os
from pathlib import Path
from persistent_store import read_bytes, write_bytes


AUTH_FILE = Path(__file__).resolve().parent / ".project_x_auth.json"
ACCESS_FILE = Path(__file__).resolve().parent / "project_x_access.json"


def public_research_enabled() -> bool:
    """Allow password-free research without exposing personal portfolio controls."""
    override = os.getenv("PROJECT_X_PUBLIC_RESEARCH")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    try:
        payload = read_bytes(ACCESS_FILE)
        return bool(json.loads(payload.decode("utf-8")).get("public_research", False)) if payload else False
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def save_public_research(enabled: bool) -> bool:
    """Persist the local access preference atomically."""
    write_bytes(ACCESS_FILE, json.dumps({"public_research": bool(enabled)}, ensure_ascii=False, indent=2).encode("utf-8"))
    return bool(enabled)


def password_is_configured() -> bool:
    return bool(os.getenv("PROJECT_X_PASSWORD")) or AUTH_FILE.exists()


def verify_password(password: str) -> bool:
    cloud_password = os.getenv("PROJECT_X_PASSWORD")
    if cloud_password:
        return hmac.compare_digest(password, cloud_password)
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(data["salt"]), int(data["iterations"])).hex()
        return hmac.compare_digest(derived, data["digest"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
