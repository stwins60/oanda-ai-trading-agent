import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
from typing import Dict, Any, List
import requests
from config import settings

logger = logging.getLogger("oanda_agent.notifier")


class Notifier:
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []

    def _record_audit(self, channel: str, title: str, message: str, level: str = "INFO"):
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "channel": channel,
            "level": level,
            "title": title,
            "message": message,
        }
        self.audit_log.insert(0, entry)
        self.audit_log = self.audit_log[:100]

    def send_discord_trade_alert(
        self,
        instrument: str,
        direction: str,
        units: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        sentiment_score: float,
        reasoning: str,
    ):
        """Dispatches a rich Discord embed for executed trades."""
        color = 0x10B981 if direction.upper() in ("BUY", "LONG") else 0xEF4444
        now_iso = datetime.now(timezone.utc).isoformat()

        embed = {
            "title": f"⚡ Trade Executed: {direction.upper()} {instrument}",
            "color": color,
            "fields": [
                {"name": "Instrument", "value": f"`{instrument}`", "inline": True},
                {"name": "Direction", "value": f"**{direction.upper()}**", "inline": True},
                {"name": "Units", "value": f"`{units}`", "inline": True},
                {"name": "Entry Price", "value": f"`{entry_price}`", "inline": True},
                {"name": "Stop-Loss", "value": f"`{sl_price}`", "inline": True},
                {"name": "Take-Profit", "value": f"`{tp_price}`", "inline": True},
                {"name": "AI Sentiment", "value": f"`{sentiment_score:+.2f}`", "inline": True},
                {"name": "AI Catalyst & Confluence Rationale", "value": reasoning[:1000], "inline": False},
            ],
            "footer": {"text": "OANDA v20 Autonomous AI Trading Agent"},
            "timestamp": now_iso,
        }

        self._record_audit(
            "DISCORD",
            f"Trade Executed: {direction} {instrument}",
            f"Units={units} @ {entry_price} | SL={sl_price} | TP={tp_price}",
            level="SUCCESS",
        )

        if not settings.DISCORD_WEBHOOK_URL:
            return

        try:
            requests.post(
                settings.DISCORD_WEBHOOK_URL,
                json={"embeds": [embed]},
                timeout=8,
            )
        except Exception as e:
            logger.error(f"Discord webhook dispatch error: {e}")

    def send_discord_trade_closed_alert(
        self, trade_id: str, instrument: str, side: str, units: int,
        entry_price: float, close_price: float, realized_pnl: float, reason: str,
    ):
        """Dispatches a Discord embed when a trade closes (TP, SL, manual or kill-switch)."""
        won = realized_pnl >= 0
        self._record_audit(
            "DISCORD",
            f"Trade Closed: #{trade_id} {instrument}",
            f"{side} {units} | {entry_price} -> {close_price} | P/L {realized_pnl:+.2f} ({reason})",
            level="SUCCESS" if won else "WARNING",
        )
        if not settings.DISCORD_WEBHOOK_URL:
            return
        embed = {
            "title": f"{'✅' if won else '❌'} Trade Closed: {side} {instrument} ({realized_pnl:+.2f})",
            "color": 0x10B981 if won else 0xEF4444,
            "fields": [
                {"name": "Trade", "value": f"`#{trade_id}`", "inline": True},
                {"name": "Units", "value": f"`{units}`", "inline": True},
                {"name": "Close Reason", "value": reason, "inline": True},
                {"name": "Entry", "value": f"`{entry_price}`", "inline": True},
                {"name": "Exit", "value": f"`{close_price}`", "inline": True},
                {"name": "Realized P/L", "value": f"**{realized_pnl:+.2f}**", "inline": True},
            ],
            "footer": {"text": "OANDA v20 Autonomous AI Trading Agent"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            requests.post(settings.DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=8)
        except Exception as e:
            logger.error(f"Discord close alert dispatch error: {e}")

    def send_system_alert(self, title: str, description: str, critical: bool = False):
        """Sends system alerts (Kill-Switch, Daily Loss Halt, Margin Warning) to Discord and SMTP Email."""
        level = "CRITICAL" if critical else "WARNING"
        self._record_audit("SYSTEM", title, description, level=level)

        if settings.DISCORD_WEBHOOK_URL:
            try:
                embed = {
                    "title": f"🚨 {title}" if critical else f"ℹ️ {title}",
                    "description": description,
                    "color": 0xDC2626 if critical else 0xF59E0B,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                requests.post(settings.DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=8)
            except Exception as e:
                logger.error(f"Discord system alert failed: {e}")

        if critical:
            self.send_email_alert(subject=f"[CRITICAL] OANDA AI Agent: {title}", html_body=f"""
                <div style="font-family: Arial, sans-serif; padding: 20px; background: #111827; color: #F9FAFB; border-radius: 8px;">
                    <h2 style="color: #EF4444;">🚨 {title}</h2>
                    <p style="font-size: 15px; line-height: 1.5;">{description}</p>
                    <hr style="border-color: #374151;" />
                    <small style="color: #9CA3AF;">Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</small>
                </div>
            """)

    def send_email_alert(self, subject: str, html_body: str):
        """Sends an HTML email alert via configured SMTP server."""
        if not (settings.SMTP_USER and settings.SMTP_PASS and settings.ALERT_RECIPIENT_EMAIL):
            return

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = settings.EMAIL_FROM
            msg["To"] = settings.ALERT_RECIPIENT_EMAIL
            msg.attach(MIMEText(html_body, "html"))

            if settings.SMTP_PORT == 465:  # Implicit TLS (e.g. Resend on 465); STARTTLS would fail here
                server_ctx = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10)
            else:
                server_ctx = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10)
            with server_ctx as server:
                if settings.SMTP_PORT != 465:
                    server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASS)
                server.sendmail(settings.EMAIL_FROM, [settings.ALERT_RECIPIENT_EMAIL], msg.as_string())

            self._record_audit("EMAIL", subject, f"Sent to {settings.ALERT_RECIPIENT_EMAIL}", level="INFO")
        except Exception as e:
            logger.error(f"SMTP Email dispatch failed: {e}")

