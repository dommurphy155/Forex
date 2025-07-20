import os
import sys
import asyncio
import aiohttp
import sqlite3
import logging
import datetime
import math
import json
import numpy as np
import xgboost as xgb
from typing import List, Dict, Any, Optional, Tuple
from filelock import FileLock
from logging.handlers import RotatingFileHandler
from aiohttp.client_exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback

# --- CONFIGURATION ---
CONFIG_FILE = os.getenv("FOREX_CONFIG", "config.json")
DEFAULT_CONFIG = {
    "PAIRS": ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "AUD_USD", "NZD_USD", "USD_CHF"],
    "RISK_PCT": 0.02,
    "LEVERAGE": 33,
    "MAX_TRADES_DAY": 20,
    "MIN_TRADES_DAY": 5,
    "MAX_CONCURRENT": 1,
    "TRADE_LIMIT_SEC": 2 * 60 * 60,  # 2 hours
    "TRADE_CHECK_INTERVAL": 15,
    "DB_FILE": "forex_bot_state.sqlite",
    "PEAK_HOURS_UTC": [(12, 16)],  # London/New York overlap
    "LOG_MAX_BYTES": 10 * 1024 * 1024,  # 10MB
    "LOG_BACKUP_COUNT": 5
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG

CONFIG = load_config()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([OANDA_API_KEY, OANDA_ACCOUNT_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    print("ERROR: Missing environment variables. Please export OANDA_API_KEY, OANDA_ACCOUNT_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
    sys.exit(1)

TELEGRAM_CHAT_ID = str(TELEGRAM_CHAT_ID)

# Logging setup with rotation
logger = logging.getLogger("forex_bot")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler = RotatingFileHandler(
    "forex_bot.log", maxBytes=CONFIG["LOG_MAX_BYTES"], backupCount=CONFIG["LOG_BACKUP_COUNT"]
)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# --- DATABASE ---
class DB:
    def __init__(self):
        os.makedirs(os.path.dirname(CONFIG["DB_FILE"]), exist_ok=True)
        self.conn = sqlite3.connect(CONFIG["DB_FILE"], detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                pair TEXT NOT NULL,
                open_time TIMESTAMP NOT NULL,
                close_time TIMESTAMP,
                direction TEXT NOT NULL,
                units INTEGER NOT NULL,
                open_price REAL NOT NULL,
                close_price REAL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                expected_roi REAL,
                pl REAL,
                duration_sec INTEGER,
                status TEXT NOT NULL CHECK(status IN ('OPEN','CLOSED'))
            )
            """)

    def insert_trade(self, trade: Dict[str, Any]):
        with self.conn:
            self.conn.execute("""
            INSERT OR IGNORE INTO trades (trade_id, pair, open_time, direction, units, open_price, stop_loss, take_profit, confidence, expected_roi, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                trade["trade_id"],
                trade["pair"],
                trade["open_time"],
                trade["direction"],
                trade["units"],
                trade["open_price"],
                trade["stop_loss"],
                trade["take_profit"],
                trade["confidence"],
                trade["expected_roi"]
            ))

    def update_trade_close(self, trade_id: str, close_price: float, close_time: datetime.datetime, pl: float, duration_sec: int, status: str):
        with self.conn:
            self.conn.execute("""
            UPDATE trades SET close_price=?, close_time=?, pl=?, duration_sec=?, status=?
            WHERE trade_id=?
            """, (close_price, close_time, pl, duration_sec, status, trade_id))

    def get_open_trades(self) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM trades WHERE status='OPEN'")
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def get_closed_trades(self, limit: int=5) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY close_time DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def count_trades_today(self) -> int:
        today = datetime.datetime.utcnow().date()
        cur = self.conn.execute("SELECT COUNT(*) FROM trades WHERE DATE(open_time)=?", (today.isoformat(),))
        return cur.fetchone()[0]

# --- OANDA CLIENT ---
class OandaClient:
    BASE_URL = "https://api-fxpractice.oanda.com/v3"

    def __init__(self, api_key: str, account_id: str):
        self.api_key = api_key
        self.account_id = account_id
        self.session = None

    async def initialize_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )

    async def close(self):
        if self.session:
            await self.session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_account_summary(self) -> Dict[str, Any]:
        if not self.session:
            await self.initialize_session()
        url = f"{self.BASE_URL}/accounts/{self.account_id}/summary"
        async with self.session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        if not self.session:
            await self.initialize_session()
        url = f"{self.BASE_URL}/accounts/{self.account_id}/openPositions"
        async with self.session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("positions", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_instruments_candles(self, instrument: str, count: int = 50, granularity: str = "M5") -> Dict[str, Any]:
        if not self.session:
            await self.initialize_session()
        url = f"{self.BASE_URL}/instruments/{instrument}/candles"
        params = {"count": count, "granularity": granularity, "price": "M"}
        async with self.session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_order(self, instrument: str, units: int, stop_loss: float, take_profit: float) -> Dict[str, Any]:
        if not self.session:
            await self.initialize_session()
        url = f"{self.BASE_URL}/accounts/{self.account_id}/orders"
        direction = "BUY" if units > 0 else "SELL"
        data = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{stop_loss:.5f}"},
                "takeProfitOnFill": {"price": f"{take_profit:.5f}"}
            }
        }
        async with self.session.post(url, json=data) as resp:
            resp.raise_for_status()
            return await resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def close_trade(self, trade_id: str) -> Dict[str, Any]:
        if not self.session:
            await self.initialize_session()
        url = f"{self.BASE_URL}/accounts/{self.account_id}/trades/{trade_id}/close"
        async with self.session.put(url) as resp:
            resp.raise_for_status()
            return await resp.json()

# --- TRADE LOGIC ---
class TradeLogic:
    PIP_VALUES = {
        "USD_JPY": 0.01,
        "EUR_USD": 0.0001,
        "GBP_USD": 0.0001,
        "USD_CAD": 0.0001,
        "AUD_USD": 0.0001,
        "NZD_USD": 0.0001,
        "USD_CHF": 0.0001
    }

    def __init__(self):
        self.model = self._train_dummy_model()

    def _train_dummy_model(self):
        X = np.array([
            [0.0010, 0.0005, 0.0008],
            [0.0020, -0.0003, 0.0012],
            [0.0015, 0.0008, 0.0010],
            [0.0018, -0.0001, 0.0013],
            [0.0009, 0.0002, 0.0007],
            [0.0021, -0.0006, 0.0014]
        ])
        y = np.array([0.02, 0.01, 0.03, 0.025, 0.018, 0.01])
        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "reg:squarederror", "verbosity": 0}
        bst = xgb.train(params, dtrain, num_boost_round=10)
        return bst

    def prepare_features(self, candles: List[Dict[str, Any]]) -> np.ndarray:
        closes = np.array([float(c["mid"]["c"]) for c in candles])
        highs = np.array([float(c["mid"]["h"]) for c in candles])
        lows = np.array([float(c["mid"]["l"]) for c in candles])

        atr = self._calc_atr(highs, lows, closes)
        price_change_pct = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
        volatility = np.std(closes[-10:]) if len(closes) >= 10 else np.std(closes)

        return np.array([atr, price_change_pct, volatility]).reshape(1, -1)

    def _calc_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return np.mean(trs) if trs else 0.0

    def score_trade(self, features: np.ndarray) -> float:
        dtest = xgb.DMatrix(features)
        pred = self.model.predict(dtest)
        return float(pred[0])

    def decide_direction(self, confidence: float) -> Optional[str]:
        if confidence < 0.5:
            return None
        return "BUY" if confidence > 0.02 else "SELL"

    def calculate_sl_tp(self, candles: List[Dict[str, Any]], entry_price: float, direction: str) -> Tuple[float, float]:
        closes = np.array([float(c["mid"]["c"]) for c in candles])
        atr = self._calc_atr(
            np.array([float(c["mid"]["h"]) for c in candles]),
            np.array([float(c["mid"]["l"]) for c in candles]),
            closes
        )
        if direction == "BUY":
            sl = max(entry_price - atr * 1.5, 0.00001)
            tp = entry_price + atr * 3.0
        else:
            sl = entry_price + atr * 1.5
            tp = max(entry_price - atr * 3.0, 0.00001)
        return sl, tp

    def calculate_pl(self, pair: str, open_price: float, close_price: float, units: int, direction: str) -> float:
        pip_value = self.PIP_VALUES.get(pair, 0.0001)
        pips = (close_price - open_price) / pip_value if direction == "BUY" else (open_price - close_price) / pip_value
        return pips * abs(units) * pip_value

# --- TELEGRAM INTERFACE ---
class TelegramBot:
    BASE_URL = "https://api.telegram.org/bot"

    def __init__(self, token: str, chat_id: str, bot_logic: Any):
        self.token = token
        self.chat_id = chat_id
        self.bot_logic = bot_logic
        self.session = None
        self.offset = None
        self.last_error = None

    async def initialize_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def clear_updates(self):
        try:
            await self.initialize_session()
            url = f"{self.BASE_URL}{self.token}/getUpdates"
            async with self.session.get(url, params={"offset": -1}) as resp:
                resp.raise_for_status()
                logger.info("Cleared Telegram updates")
        except Exception as e:
            logger.error(f"Failed to clear Telegram updates: {e}")

    async def close(self):
        if self.session:
            await self.session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_updates(self) -> List[Dict[str, Any]]:
        await self.initialize_session()
        url = f"{self.BASE_URL}{self.token}/getUpdates"
        params = {"timeout": 20}
        if self.offset is not None:
            params["offset"] = self.offset
        async with self.session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if not data.get("ok"):
                return []
            return data.get("result", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_message(self, text: str):
        await self.initialize_session()
        url = f"{self.BASE_URL}{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        async with self.session.post(url, json=payload) as resp:
            resp.raise_for_status()

    async def handle_commands(self):
        try:
            updates = await self.get_updates()
            for update in updates:
                self.offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue
                from_id = str(message.get("from", {}).get("id", ""))
                if from_id != self.chat_id:
                    await self.send_message("Unauthorized user. Access denied.")
                    continue
                text = message.get("text", "").strip()
                if text.startswith("/"):
                    cmd = text.split()[0].lower()
                    handler = getattr(self, f"cmd_{cmd[1:]}", None)
                    if handler:
                        await handler()
                    else:
                        await self.send_message(f"Unknown command: {cmd}")
        except Exception as e:
            self.last_error = traceback.format_exc()
            logger.error(f"Error handling Telegram commands: {e}\n{self.last_error}")

    async def cmd_daily(self):
        trades_today = self.bot_logic.db.count_trades_today()
        closed = self.bot_logic.db.get_closed_trades(limit=1000)
        pl_total = sum(t.get("pl") or 0 for t in closed if t.get("close_time") and t.get("close_time").date() == datetime.datetime.utcnow().date())
        roi_pct = (pl_total / self.bot_logic.balance) * 100 if self.bot_logic.balance else 0
        await self.send_message(f"Today's summary:\nTrades: {trades_today}\nP/L: £{pl_total:.2f}\nROI: {roi_pct:.2f}%")

    async def cmd_weekly(self):
        one_week_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
        closed = self.bot_logic.db.get_closed_trades(limit=1000)
        closed_week = [t for t in closed if t.get("close_time") and t.get("close_time") > one_week_ago]
        pl_total = sum(t.get("pl") or 0 for t in closed_week)
        best_trade = max(closed_week, key=lambda t: t.get("pl") or -math.inf, default=None)
        best_info = f"{best_trade['pair']} +£{best_trade['pl']:.2f}" if best_trade else "N/A"
        await self.send_message(f"Weekly ROI: £{pl_total:.2f}\nBest trade: {best_info}")

    async def cmd_status(self):
        await self.send_message(f"Bot is running. Balance: £{self.bot_logic.balance:.2f}, Open trades: {len(self.bot_logic.open_trades)}")

    async def cmd_maketrade(self):
        trade = await self.bot_logic.auto_place_trade()
        if trade:
            msg = (
                f"Trade placed:\nPair: {trade['pair']}\nUnits: {trade['units']}\n"
                f"Expected ROI: {trade['expected_roi']*100:.2f}%\n"
                f"Stop Loss: {trade['stop_loss']:.5f}\nTake Profit: {trade['take_profit']:.5f}"
            )
            await self.send_message(msg)
        else:
            await self.send_message("No suitable trade to place now.")

    async def cmd_open(self):
        open_trades = self.bot_logic.db.get_open_trades()
        if not open_trades:
            await self.send_message("No open trades.")
            return
        msg = "Open trades:\n"
        for t in open_trades:
            msg += (f"{t['pair']} {t['direction']} units:{t['units']} opened at {t['open_time']} "
                    f"SL: {t['stop_loss']:.5f} TP: {t['take_profit']:.5f}\n")
        await self.send_message(msg)

    async def cmd_closed(self):
        closed = self.bot_logic.db.get_closed_trades()
        if not closed:
            await self.send_message("No closed trades.")
            return
        msg = "Last 5 closed trades:\n"
        for t in closed:
            duration = t['duration_sec'] or 0
            pl = t.get('pl') or 0
            msg += (f"{t['pair']} {t['direction']} P/L: £{pl:.2f} Duration: {duration//60}m\n")
        await self.send_message(msg)

    async def cmd_debug(self):
        if self.last_error:
            await self.send_message(f"Last error:\n{self.last_error[:1000]}")
        else:
            await self.send_message("No recent errors recorded.")

# --- MAIN BOT LOGIC ---
class ForexBot:
    def __init__(self):
        self.db = DB()
        self.oanda = OandaClient(OANDA_API_KEY, OANDA_ACCOUNT_ID)
        self.trade_logic = TradeLogic()
        self.telegram = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, self)
        self.balance = 0.0
        self.open_trades = []
        self.trade_count_today = 0
        self.lock_file = "/tmp/forex_bot.lock"

    async def initialize(self):
        await self.oanda.initialize_session()
        await self.telegram.initialize_session()
        await self.telegram.clear_updates()

    async def update_account_balance(self):
        try:
            data = await self.oanda.get_account_summary()
            self.balance = float(data["account"]["balance"])
            logger.info(f"Account balance updated: £{self.balance:.2f}")
        except ClientError as e:
            logger.error(f"Failed to update account balance: {e}")

    async def refresh_open_trades(self):
        try:
            positions = await self.oanda.get_open_positions()
            self.open_trades = []
            for pos in positions:
                if pos["instrument"] not in CONFIG["PAIRS"]:
                    continue
                long_units = int(pos.get("long", {}).get("units", 0))
                short_units = int(pos.get("short", {}).get("units", 0))
                if long_units == 0 and short_units == 0:
                    logger.debug(f"Skipping position for {pos['instrument']}: no units")
                    continue
                units = long_units if long_units != 0 else -short_units
                direction = "BUY" if long_units != 0 else "SELL"
                average_price = pos.get("long", {}).get("averagePrice") or pos.get("short", {}).get("averagePrice")
                if not average_price:
                    logger.error(f"No averagePrice for {pos['instrument']}: {pos}")
                    continue
                trade = {
                    "trade_id": pos.get("tradeIDs", [None])[0],
                    "pair": pos["instrument"],
                    "units": units,
                    "open_price": float(average_price),
                    "direction": direction,
                    "open_time": datetime.datetime.utcnow()
                }
                self.open_trades.append(trade)
            logger.info(f"Open trades refreshed: {len(self.open_trades)}")
        except ClientError as e:
            logger.error(f"Failed to refresh open trades: {e}\n{traceback.format_exc()}")

    async def auto_place_trade(self) -> Optional[Dict[str, Any]]:
        if len(self.open_trades) >= CONFIG["MAX_CONCURRENT"]:
            logger.info("Max concurrent trades reached.")
            return None

        trades_today = self.db.count_trades_today()
        if trades_today >= CONFIG["MAX_TRADES_DAY"]:
            logger.info("Max trades for today reached.")
            return None

        best_candidate = None
        best_score = 0.0
        best_data = None

        for pair in CONFIG["PAIRS"]:
            try:
                candles = await self.oanda.get_instruments_candles(pair, count=50)
                if "candles" not in candles:
                    logger.warning(f"No candles for {pair}")
                    continue
                features = self.trade_logic.prepare_features(candles["candles"])
                confidence = self.trade_logic.score_trade(features)

                if confidence < 0.5:
                    continue
                expected_roi = confidence

                if expected_roi < 0.01:
                    continue

                now_utc = datetime.datetime.utcnow().hour
                in_peak = any(start <= now_utc < end for start, end in CONFIG["PEAK_HOURS_UTC"])
                if not in_peak and not (confidence == 1.0 and expected_roi >= 0.10):
                    continue

                if confidence > best_score:
                    best_score = confidence
                    best_candidate = pair
                    best_data = (candles["candles"], confidence, expected_roi)
            except ClientError as e:
                logger.error(f"Failed to process pair {pair}: {e}")

        if not best_candidate:
            return None

        candles, confidence, expected_roi = best_data
        entry_price = float(candles[-1]["mid"]["c"])
        direction = self.trade_logic.decide_direction(confidence)
        if direction is None:
            return None

        atr = self.trade_logic._calc_atr(
            np.array([float(c["mid"]["h"]) for c in candles]),
            np.array([float(c["mid"]["l"]) for c in candles]),
            np.array([float(c["mid"]["c"]) for c in candles])
        )
        risk_amount = self.balance * CONFIG["RISK_PCT"]
        pip_value = self.trade_logic.PIP_VALUES.get(best_candidate, 0.0001)
        units = int((risk_amount / (atr / pip_value)) * CONFIG["LEVERAGE"])
        if direction == "SELL":
            units = -units

        stop_loss, take_profit = self.trade_logic.calculate_sl_tp(candles, entry_price, direction)

        try:
            order_resp = await self.oanda.create_order(best_candidate, units, stop_loss, take_profit)
            order_id = order_resp["orderFillTransaction"]["id"]
        except ClientError as e:
            logger.error(f"Failed to place order: {e}")
            return None

        trade = {
            "trade_id": order_id,
            "pair": best_candidate,
            "open_time": datetime.datetime.utcnow(),
            "direction": direction,
            "units": units,
            "open_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "expected_roi": expected_roi,
            "status": "OPEN"
        }
        self.db.insert_trade(trade)
        self.open_trades.append(trade)
        logger.info(f"Trade placed: {trade}")
        return trade

    async def close_profitable_trades(self):
        open_trades = self.db.get_open_trades()
        for t in open_trades:
            open_time = t["open_time"]
            duration = (datetime.datetime.utcnow() - open_time).total_seconds()
            if duration < CONFIG["TRADE_LIMIT_SEC"]:
                continue

            try:
                candles = await self.oanda.get_instruments_candles(t["pair"], count=1)
                if "candles" not in candles or not candles["candles"]:
                    logger.warning(f"No candles for {t['pair']} when closing trade")
                    continue
                current_price = float(candles["candles"][-1]["mid"]["c"])

                pl = self.trade_logic.calculate_pl(t["pair"], t["open_price"], current_price, t["units"], t["direction"])

                if pl <= 0:
                    continue

                await self.oanda.close_trade(t["trade_id"])
                close_time = datetime.datetime.utcnow()
                duration_sec = int((close_time - open_time).total_seconds())
                self.db.update_trade_close(t["trade_id"], current_price, close_time, pl, duration_sec, "CLOSED")
                logger.info(f"Closed trade {t['trade_id']} profitably after {duration_sec}s with P/L: £{pl:.2f}")
                self.open_trades = [ot for ot in self.open_trades if ot["trade_id"] != t["trade_id"]]
            except ClientError as e:
                logger.error(f"Failed to close trade {t['trade_id']}: {e}")

    async def run(self):
        logger.info("ForexBot started")
        await self.initialize()
        try:
            while True:
                try:
                    await self.update_account_balance()
                    await self.refresh_open_trades()
                    await self.close_profitable_trades()
                    await self.telegram.handle_commands()
                except Exception as e:
                    self.telegram.last_error = traceback.format_exc()
                    logger.error(f"Unhandled exception in main loop: {e}\n{self.telegram.last_error}")
                await asyncio.sleep(CONFIG["TRADE_CHECK_INTERVAL"])
        finally:
            await self.oanda.close()
            await self.telegram.close()

if __name__ == "__main__":
    lock = FileLock("/tmp/forex_bot.lock", timeout=1)
    try:
        with lock:
            bot = ForexBot()
            asyncio.run(bot.run())
    except FileLock.Timeout:
        logger.error("Another instance of the bot is running. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)
