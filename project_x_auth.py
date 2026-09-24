"""Local password verification for the externally exposed Project X app."""

import hashlib
import hmac
import json
import os
from pathlib import Path


AUTH_FILE = Path(__file__).resolve().parent / ".project_x_auth.json"


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
