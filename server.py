import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from auth import SESSION_COOKIE, sessions, client_ip, require_auth, websocket_authenticated
from broker import OandaBroker
from news_sentiment import NewsSentimentEngine
from notifier import Notifier
from agent import AutonomousTradingAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oanda_agent.server")

broker = OandaBroker()
sentiment_engine = NewsSentimentEngine()
notifier = Notifier()
trading_agent = AutonomousTradingAgent(broker, sentiment_engine, notifier)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, payload: dict):
        for conn in list(self.active_connections):
            if not websocket_authenticated(conn):  # Session logged out or expired since connecting
                self.disconnect(conn)
                await conn.close(code=1008)
                continue
            try:
                await conn.send_json(payload)
            except Exception:
                self.disconnect(conn)


ws_manager = ConnectionManager()


def build_dashboard_state() -> dict:
    account = broker.get_account_summary()
    open_trades = broker.get_open_trades()
    return {
        "type": "STATE_UPDATE",
        "is_running": trading_agent.is_running,
        "daily_halt_triggered": trading_agent.daily_halt_triggered,
        "watchlist": settings.WATCHLIST,
        "account": account,
        "open_trades": open_trades,
        "technicals": trading_agent.latest_technicals,
        "sentiment": trading_agent.latest_sentiment,
        "ai_decisions": trading_agent.latest_ai,
        "strategy_mode": settings.STRATEGY_MODE,
        "decision_log": trading_agent.decision_log[:35],
        "audit_log": notifier.audit_log[:35],
    }


async def background_agent_loop():
    """Periodically runs the trading agent evaluation cycle and broadcasts state via WebSocket."""
    while True:
        try:
            await asyncio.to_thread(trading_agent.run_cycle)
            state = await asyncio.to_thread(build_dashboard_state)
            await ws_manager.broadcast(state)
        except Exception as e:
            logger.error(f"Background loop error: {e}")
        await asyncio.sleep(settings.POLLING_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_agent_loop())
    yield
    task.cancel()


app = FastAPI(title="OANDA v20 Autonomous AI Trading Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Every /api route except login requires an authenticated operator session
api = APIRouter(dependencies=[Depends(require_auth)])

FRONTEND_DIR = Path(__file__).parent / "frontend"


class LoginRequest(BaseModel):
    username: str
    password: str


class ManualOrderRequest(BaseModel):
    instrument: str
    direction: str  # "BUY" or "SELL"
    units: int
    stop_loss_pips: float = 20.0
    take_profit_pips: float = 40.0


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    if not sessions.is_valid(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse(content=(FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    if sessions.is_valid(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(content=(FRONTEND_DIR / "login.html").read_text(encoding="utf-8"))


@app.post("/api/auth/login")
async def login(req: LoginRequest, request: Request, response: Response):
    ip = client_ip(request)
    if sessions.is_locked_out(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 15 minutes.")
    if not sessions.verify_credentials(req.username, req.password, ip):
        await asyncio.sleep(1)  # Slow down brute-force guessing
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    response.set_cookie(
        SESSION_COOKIE,
        sessions.create_session(),
        max_age=settings.SESSION_TTL_HOURS * 3600,
        httponly=True,
        samesite="strict",
        secure=settings.COOKIE_SECURE,
    )
    notifier._record_audit("AUTH", "Operator Login", f"Dashboard login from {ip}", "INFO")
    return {"success": True}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    sessions.revoke(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)
    return {"success": True}


@api.get("/api/state")
async def get_state():
    return await asyncio.to_thread(build_dashboard_state)


@api.get("/api/performance")
async def get_performance():
    """Closed-trade statistics (win rate, realized P/L, profit factor) from OANDA history."""
    perf = await asyncio.to_thread(trading_agent.performance_summary)
    account = await asyncio.to_thread(broker.get_account_summary)
    perf["account_balance"] = account.get("balance", 0.0)
    perf["account_lifetime_pl"] = account.get("lifetime_pl", 0.0)
    return perf


@api.get("/api/candles/{instrument}")
async def get_candles(instrument: str, granularity: str = "M15", count: int = 200):
    instrument = instrument.upper()
    if instrument not in settings.WATCHLIST:
        raise HTTPException(status_code=400, detail=f"{instrument} is not in the configured watchlist.")
    df = await asyncio.to_thread(broker.get_candles, instrument, granularity, count)
    if df.empty:
        return {"instrument": instrument, "granularity": granularity, "candles": []}
    return {
        "instrument": instrument,
        "granularity": granularity,
        "candles": df.to_dict(orient="records"),
    }


@api.post("/api/bot/toggle")
async def toggle_bot():
    trading_agent.is_running = not trading_agent.is_running
    status_str = "STARTED" if trading_agent.is_running else "PAUSED"
    notifier.send_system_alert(
        f"Autonomous Bot {status_str}",
        f"Operator toggled automated execution to: {status_str}.",
        critical=False,
    )
    state = await asyncio.to_thread(build_dashboard_state)
    await ws_manager.broadcast(state)
    return {"is_running": trading_agent.is_running}


@api.post("/api/bot/scan-now")
async def trigger_immediate_scan():
    await asyncio.to_thread(trading_agent.run_cycle)
    state = await asyncio.to_thread(build_dashboard_state)
    await ws_manager.broadcast(state)
    return state


@api.post("/api/trades/manual")
async def place_manual_trade(req: ManualOrderRequest):
    req.instrument = req.instrument.upper()
    if req.instrument not in settings.WATCHLIST:
        raise HTTPException(status_code=400, detail=f"{req.instrument} is not in the configured watchlist.")

    df = await asyncio.to_thread(broker.get_candles, req.instrument, "M5", 5)
    if df.empty:
        raise HTTPException(status_code=400, detail="Could not fetch current price for instrument.")

    current_price = float(df.iloc[-1]["close"])
    pip_value = 0.01 if "JPY" in req.instrument else 0.0001

    if req.direction.upper() == "BUY":
        signed_units = abs(req.units)
        sl_price = current_price - (req.stop_loss_pips * pip_value)
        tp_price = current_price + (req.take_profit_pips * pip_value)
    else:
        signed_units = -abs(req.units)
        sl_price = current_price + (req.stop_loss_pips * pip_value)
        tp_price = current_price - (req.take_profit_pips * pip_value)

    res = await asyncio.to_thread(
        broker.place_market_order, req.instrument, signed_units, sl_price, tp_price
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=str(res.get("error")))

    notifier.send_discord_trade_alert(
        instrument=req.instrument,
        direction=req.direction.upper(),
        units=signed_units,
        entry_price=res.get("fill_price") or current_price,
        sl_price=float(res["stop_loss"]),
        tp_price=float(res["take_profit"]),
        sentiment_score=0.0,
        reasoning="Manual operator override trade from Web Dashboard.",
    )
    state = await asyncio.to_thread(build_dashboard_state)
    await ws_manager.broadcast(state)
    return res


@api.post("/api/trades/{trade_id}/close")
async def close_single_trade(trade_id: str):
    res = await asyncio.to_thread(broker.close_trade, trade_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=str(res.get("error")))
    notifier._record_audit("MANUAL", f"Closed Trade #{trade_id}", "Operator manually closed position.", "INFO")
    state = await asyncio.to_thread(build_dashboard_state)
    await ws_manager.broadcast(state)
    return res


@api.post("/api/bot/kill-switch")
async def emergency_kill_switch():
    """Halts automated trading immediately and liquidates all open positions on OANDA."""
    trading_agent.is_running = False
    res = await asyncio.to_thread(broker.close_all_positions)
    notifier.send_system_alert(
        "EMERGENCY KILL-SWITCH ACTIVATED",
        f"Automated trading halted. Closed instruments: {res.get('closed_instruments')}",
        critical=True,
    )
    state = await asyncio.to_thread(build_dashboard_state)
    await ws_manager.broadcast(state)
    return res


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not websocket_authenticated(websocket):
        await websocket.close(code=1008)  # Policy violation: no valid session
        return
    await ws_manager.connect(websocket)
    try:
        initial_state = await asyncio.to_thread(build_dashboard_state)
        await websocket.send_json(initial_state)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


app.include_router(api)
