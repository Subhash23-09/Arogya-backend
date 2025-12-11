# healthbackend/services/user_profile_store.py
import json
import os

# Base dir: .../healthbackend
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
FILE = os.path.join(STORAGE_DIR, "user_profiles.json")


def _load():
    """Return dict of user profiles; empty dict if file missing."""
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data):
    """Persist profiles dict, creating folder if needed."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_profile(user_id: str, profile: dict):
    data = _load()
    data[user_id] = profile
    _save(data)


def get_profile(user_id: str) -> dict:
    return _load().get(user_id, {})
