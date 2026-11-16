import hashlib
import hmac
import logging
import secrets
import time
from typing import Dict, List, Optional
from fastapi import HTTPException, Request, WebSocket
from config import settings
from db import SessionStore

logger = logging.getLogger("oanda_agent.auth")

SESSION_COOKIE = "oanda_session"
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW_SECONDS = 15 * 60


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class SessionManager:
    """
    Single-operator login. Sessions live in memory and are mirrored to Postgres (token hashes only),
    so a server restart no longer logs the operator out. Without a database it falls back to memory-only.
    """

    def __init__(self):
        self._sessions: Dict[str, float] = {}  # token -> expiry epoch
        self._store = SessionStore()
        self._failed_attempts: Dict[str, List[float]] = {}  # client ip -> failure timestamps
        if not settings.DASHBOARD_PASSWORD:
            logger.error("DASHBOARD_PASSWORD is not set in .env; dashboard login is disabled until it is.")

    def _recent_failures(self, client_ip: str) -> List[float]:
        cutoff = time.time() - LOCKOUT_WINDOW_SECONDS
        recent = [t for t in self._failed_attempts.get(client_ip, []) if t > cutoff]
        self._failed_attempts[client_ip] = recent
        return recent

    def is_locked_out(self, client_ip: str) -> bool:
        return len(self._recent_failures(client_ip)) >= MAX_FAILED_ATTEMPTS

    def verify_credentials(self, username: str, password: str, client_ip: str) -> bool:
        if not settings.DASHBOARD_PASSWORD:
            return False
        # Compare both fields in constant time so neither leaks via timing
        user_ok = hmac.compare_digest(username.encode(), settings.DASHBOARD_USERNAME.encode())
        pass_ok = hmac.compare_digest(password.encode(), settings.DASHBOARD_PASSWORD.encode())
        if user_ok and pass_ok:
            self._failed_attempts.pop(client_ip, None)
            return True
        self._recent_failures(client_ip).append(time.time())
        logger.warning(f"Failed dashboard login attempt from {client_ip}")
        return False

    def create_session(self) -> str:
        token = secrets.token_urlsafe(32)
        expiry = time.time() + settings.SESSION_TTL_HOURS * 3600
        self._sessions[token] = expiry
        self._store.save(_hash(token), expiry)
        return token

    def is_valid(self, token: Optional[str]) -> bool:
        if not token:
            return False
        expiry = self._sessions.get(token)
        if expiry is None:
            expiry = self._store.lookup(_hash(token))  # Issued before the last restart?
            if expiry is None:
                return False
            self._sessions[token] = expiry
        if expiry < time.time():
            self._sessions.pop(token, None)
            self._store.delete(_hash(token))
            return False
        return True

    def revoke(self, token: Optional[str]):
        if token:
            self._sessions.pop(token, None)
            self._store.delete(_hash(token))


sessions = SessionManager()


def client_ip(request: Request) -> str:
    # Behind Cloudflare Tunnel every request arrives from cloudflared; the real client IP is in a header
    if settings.TRUSTED_CLIENT_IP_HEADER:
        forwarded = request.headers.get(settings.TRUSTED_CLIENT_IP_HEADER)
        if forwarded:
            return forwarded.strip()
    return request.client.host if request.client else "unknown"


def require_auth(request: Request):
    """FastAPI dependency: rejects requests without a valid session cookie."""
    if not sessions.is_valid(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(status_code=401, detail="Not authenticated")


def websocket_authenticated(websocket: WebSocket) -> bool:
    return sessions.is_valid(websocket.cookies.get(SESSION_COOKIE))
