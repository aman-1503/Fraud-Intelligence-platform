"""JWT bearer auth. Reviewer identity comes from the token, not the request body."""
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings

_PLACEHOLDERS = {
    "",
    "your-secret-key-change-in-production",
    "your-production-secret-change-this",
}
bearer = HTTPBearer(auto_error=False)


def secret_configured() -> bool:
    return settings.JWT_SECRET not in _PLACEHOLDERS


def assert_secret_configured():
    if not secret_configured():
        raise RuntimeError("JWT_SECRET is missing or still a placeholder")


def create_access_token(subject: str, role: str) -> str:
    assert_secret_configured()
    if role not in {"scorer", "reviewer", "admin"}:
        raise ValueError("role must be scorer, reviewer, or admin")
    payload = {
        "sub": subject,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def require_roles(*roles: str):
    async def dependency(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if creds is None or not secret_configured():
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            payload = jwt.decode(creds.credentials, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        except JWTError:
            raise HTTPException(status_code=401, detail="Not authenticated")
        role = payload.get("role")
        if role != "admin" and role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Not authenticated")
        return payload

    return dependency
