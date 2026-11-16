import os
from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # OANDA Credentials
    OANDA_ACCESS_TOKEN: str = os.getenv("OANDA_ACCESS_TOKEN", "")
    OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")
    OANDA_ENVIRONMENT: str = os.getenv("OANDA_ENVIRONMENT", "practice").lower()

    # LLM & News Credentials
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # Notifications
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASS: str = os.getenv("SMTP_PASS", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "") or os.getenv("SMTP_USER", "")
    ALERT_RECIPIENT_EMAIL: str = os.getenv("EMAIL_TO", "") or os.getenv("ALERT_RECIPIENT_EMAIL", "")

    # Dashboard Login
    DASHBOARD_USERNAME: str = os.getenv("DASHBOARD_USERNAME", "admin")
    DASHBOARD_PASSWORD: str = os.getenv("DASHBOARD_PASSWORD", "")
    SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "12"))
    # Set true when serving the dashboard over HTTPS so the session cookie is never sent in clear text
    COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"

    # Header carrying the real client IP when behind a trusted proxy (e.g. "CF-Connecting-IP" for Cloudflare Tunnel).
    # Leave empty when the app is reachable directly, otherwise clients could spoof their IP.
    TRUSTED_CLIENT_IP_HEADER: str = os.getenv("TRUSTED_CLIENT_IP_HEADER", "")

    # Postgres cache for LLM results (optional; caching is disabled when unset)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    LLM_CACHE_TTL_MINUTES: int = int(os.getenv("LLM_CACHE_TTL_MINUTES", "30"))

    # Strategy & Risk Tunables
    WATCHLIST_RAW: str = os.getenv("WATCHLIST", "EUR_USD,GBP_USD,USD_JPY")
    RISK_PER_TRADE_PCT: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
    POLLING_INTERVAL_SECONDS: int = int(os.getenv("POLLING_INTERVAL_SECONDS", "60"))
    MIN_SENTIMENT_THRESHOLD: float = float(os.getenv("MIN_SENTIMENT_THRESHOLD", "0.30"))
    ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))
    ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", "2.5"))

    # Entry, confirmation and trend timeframes (lowest first), e.g. "M15,H1,H4" or "H1,H4,D"
    STRATEGY_TIMEFRAMES_RAW: str = os.getenv("STRATEGY_TIMEFRAMES", "M15,H1,H4")

    # Strategy engine: "rules" (technicals + sentiment), "ai_shadow" (AI decides, journaled only, never trades), "ai"
    STRATEGY_MODE: str = os.getenv("STRATEGY_MODE", "rules").lower()
    AI_TRADER_MODEL: str = os.getenv("AI_TRADER_MODEL", "")  # Defaults to LLM_MODEL
    AI_MIN_CONFIDENCE: float = float(os.getenv("AI_MIN_CONFIDENCE", "0.65"))

    # Daily P/L summary to Discord at the UTC day rollover
    DAILY_SUMMARY_ENABLED: bool = os.getenv("DAILY_SUMMARY_ENABLED", "true").lower() == "true"

    # Execution & trade-management guards
    MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "3"))
    # Skip entries when the live spread exceeds this fraction of the M15 ATR (wide spreads eat small edges)
    MAX_SPREAD_ATR_RATIO: float = float(os.getenv("MAX_SPREAD_ATR_RATIO", "0.30"))
    # Never commit more than this share of free margin to a single new trade
    MAX_MARGIN_USAGE_PCT: float = float(os.getenv("MAX_MARGIN_USAGE_PCT", "0.5"))
    # Move SL to break-even (+ BREAKEVEN_LOCK_R of the initial risk) once profit reaches this many R; 0 disables
    BREAKEVEN_TRIGGER_R: float = float(os.getenv("BREAKEVEN_TRIGGER_R", "1.0"))
    BREAKEVEN_LOCK_R: float = float(os.getenv("BREAKEVEN_LOCK_R", "0.1"))

    @property
    def WATCHLIST(self) -> List[str]:
        return [pair.strip().upper() for pair in self.WATCHLIST_RAW.split(",") if pair.strip()]

    @property
    def STRATEGY_TIMEFRAMES(self) -> List[str]:
        tfs = [tf.strip().upper() for tf in self.STRATEGY_TIMEFRAMES_RAW.split(",") if tf.strip()]
        if len(tfs) != 3:
            raise ValueError("STRATEGY_TIMEFRAMES must list exactly 3 granularities, e.g. M15,H1,H4")
        return tfs

    @property
    def OANDA_BASE_URL(self) -> str:
        if self.OANDA_ENVIRONMENT == "live":
            return "https://api-fxtrade.oanda.com/v3"
        return "https://api-fxpractice.oanda.com/v3"


settings = Settings()

