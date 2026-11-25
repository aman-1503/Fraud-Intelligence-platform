"""Hash every field that changes a score. The transaction id alone is not a key."""
import hashlib
import json

_FIELDS = (
    "transaction_id", "user_id", "amount", "currency",
    "merchant_category", "country", "device_id", "timestamp",
)


def request_hash(payload: dict) -> str:
    canonical = {}
    for key in _FIELDS:
        value = payload.get(key)
        if key == "amount" and value is not None:
            value = round(float(value), 2)
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        canonical[key] = value
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
