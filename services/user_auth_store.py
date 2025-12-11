# healthbackend/services/user_auth_store.py
import os
import json

# Base dir: .../healthbackend
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
FILE = os.path.join(STORAGE_DIR, "users.json")


def _load_users():
    """Return list of users; empty list if file missing."""
    if not os.path.exists(FILE):
        return []
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users):
    """Persist users list to JSON, creating folder if needed."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def check_credentials(username: str, password: str) -> bool:
    users = _load_users()
    for u in users:
        if u.get("username") == username and u.get("password") == password:
            return True
    return False


def create_user(username: str, password: str) -> bool:
    users = _load_users()
    if any(u.get("username") == username for u in users):
        return False
    users.append({"username": username, "password": password})
    _save_users(users)
    return True
