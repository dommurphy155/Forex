#!/usr/bin/env python3
import os
import sys
import time
import math
import json
import asyncio
import logging
import sqlite3
import datetime
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import xgboost as xgb

# === CONFIG ===
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([OANDA_API_KEY, OANDA_ACCOUNT_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    print("ERROR: Missing required environment variables: OANDA_API_KEY, OANDA_ACCOUNT_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
    sys.exit(1)

try:
    TELEGRAM_CHAT_ID = int(TELEGRAM_CHAT_ID)
except Exception:
    print("ERROR: TELEGRAM_CHAT_ID must be an integer")
    sys.exit(1)

OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"
PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CAD', 'AUD_USD', 'NZD_USD', 'USD_CHF']
RISK_PCT = 0.02
MAX_TRADES_DAY = 20
MIN_TRADES_DAY = 5
MAX_CONCURRENT = 1
TRADE_LIMIT_SEC = 7200  # 2 hours max trade duration
LEVERAGE = 33
PEAK_HOURS_UTC = [(12, 16)]  # London/NY overlap approx UTC 12:00-16:00
DB_FILE = "forex_bot_state.sqlite3"
LOG_FILE = "forex_bot.log"
TRADE_CHECK_INTERVAL = 15  # seconds

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("forex_bot")

# === DATABASE ===
class DB:
    def __init__(self, path=DB_FILE):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._create()

    def _create(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                pair TEXT,
                open_time TEXT,
                close_time TEXT,
                direction TEXT,
                units INTEGER,
                open_price REAL,
                close_price REAL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                expected_roi REAL,
                pl REAL,
                duration_sec INTEGER,
                status TEXT
            )
        """)
        self.conn.commit()

    def insert_trade(self, trade: Dict[str, Any]):
        c = self.conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO trades (
                trade_id, pair, open_time, direction, units, open_price, stop_loss,
                take_profit, confidence, expected_roi, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["trade_id"], trade["pair"], trade["open_time"], trade["direction"],
            trade["units"], trade["open_price"], trade["stop_loss"], trade["take_profit"],
            trade["confidence"], trade["expected_roi"], trade["status"]
        ))
        self.conn.commit()

    def update_trade_close(self, trade_id: str, close_price: float, close_time: str, pl: float, duration_sec: int, status: str):
        c = self.conn.cursor()
        c.execute("""
            UPDATE trades SET
                close_price=?, close_time=?, pl=?, duration_sec=?, status=?
            WHERE trade_id=?
        """, (close_price, close_time, pl, duration_sec, status, trade_id))
        self.conn.commit()

    def get_open_trades(self) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM trades WHERE status='OPEN'")
        rows = c.fetchall()
        keys = [desc[0] for desc in c.description]
        return [dict(zip(keys, row)) for row in rows]

    def get_closed_trades(self, limit=5) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY close_time DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        keys = [desc[0] for desc in c.description]
        return [dict(zip(keys, row)) for row in rows]

    def count_trades_today(self) -> int:
        c = self.conn.cursor()
        today = datetime.datetime.utcnow().date().isoformat()
        c.execute("SELECT COUNT(*) FROM trades WHERE open_time>=? AND status IN ('OPEN','CLOSED')", (today,))
        (cnt,) = c.fetchone()
        return cnt or 0

    def sum_pl_today(self) -> float:
        c = self.conn.cursor()
        today = datetime.datetime.utcnow().date().isoformat()
        c.execute("SELECT SUM(pl) FROM trades WHERE close_time>=? AND status='CLOSED'", (today,))
        res = c.fetchone()[0]
        return res if res is not None else 0.0

    def get_best_trade_week(self) -> Optional[Dict[str, Any]]:
        c = self.conn.cursor()
        week_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat()
        c.execute("SELECT * FROM trades WHERE close_time>=? AND status='CLOSED' ORDER BY pl DESC LIMIT 1", (week_ago,))
        row = c.fetchone()
        if not row:
            return None
        keys = [desc[0] for desc in c.description]
        return dict(zip(keys, row))


# === OANDA CLIENT ===
class OandaClient:
    def __init__(self, api_key: str, account_id: str):
        self.api_key = api_key
        self.account_id = account_id
        self.session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {api_key}"})

    async def _request(self, method: str, url: str, params=None, json_data=None):
        for attempt in range(3):
            try:
                async with self.session.request(method, url, params=params, json=json_data, timeout=15) as resp:
                    text = await resp.text()
                    if resp.status not in (200, 201):
                        logger.warning(f"OANDA {method} {url} failed {resp.status} {text}")
                        await asyncio.sleep(1)
                        continue
                    return json.loads(text)
            except Exception as e:
                logger.warning(f"OANDA {method} Exception {e}")
                await asyncio.sleep(1)
        raise RuntimeError(f"OANDA {method} {url} failed 3 times")

    async def get_account_summary(self):
        url = f"{OANDA_API_URL}/accounts/{self.account_id}/summary"
        return await self._request("GET", url)

    async def get_instruments_candles(self, instrument: str, count=50, granularity="M5"):
        url = f"{OANDA_API_URL}/instruments/{instrument}/candles"
        params = {"count": count, "granularity": granularity, "price": "M"}
        return await self._request("GET", url, params=params)

    async def get_open_positions(self):
        url = f"{OANDA_API_URL}/accounts/{self.account_id}/openPositions"
        return await self._request("GET", url)

    async def create_order(self, pair: str, units: int, sl: float, tp: float):
        url = f"{OANDA_API_URL}/accounts/{self.account_id}/orders"
        data = {
            "order": {
                "instrument": pair,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{sl}"},
                "takeProfitOnFill": {"price": f"{tp}"}
            }
        }
        return await self._request("POST", url, json_data=data)

    async def close_trade(self, trade_id: str):
        url = f"{OANDA_API_URL}/accounts/{self.account_id}/trades/{trade_id}/close"
        return await self._request("PUT", url, json_data={})

    async def close(self):
        await self.session.close()


# === TRADE LOGIC ===
class TradeLogic:
    def __init__(self):
        # Create a simple XGBoost dummy model on init to avoid file dependencies
        self.model = self._create_dummy_model()

    def _create_dummy_model(self):
        dtrain = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
        params = {'objective': 'reg:squarederror'}
        return xgb.train(params, dtrain, num_boost_round=2)

    def prepare_features(self, candles: List[Dict[str, Any]]) -> np.ndarray:
        closes = np.array([float(c["mid"]["c"]) for c in candles])
        highs = np.array([float(c["mid"]["h"]) for c in candles])
        lows = np.array([float(c["mid"]["l"]) for c in candles])
        returns = (closes[-1] - closes[:-1]) / closes[:-1] if len(closes) > 1 else np.array([0])
        atr = np.mean(highs - lows)
        vol = np.std(returns)
        feat = np.array([
            returns[-1] if len(returns) > 0 else 0,
            atr,
            vol,
            closes[-1],
            closes[-1] - closes[0]
        ])
        return feat.reshape(1, -1)

    def score_trade(self, features: np.ndarray) -> float:
        pred = self.model.predict(xgb.DMatrix(features))
        return float(pred[0])

    def calculate_sl_tp(self, candles: List[Dict[str, Any]], entry_price: float, direction: str) -> Tuple[float, float]:
        highs = np.array([float(c["mid"]["h"]) for c in candles])
        lows = np.array([float(c["mid"]["l"]) for c in candles])
        atr = np.mean(highs - lows)
        if direction == "BUY":
            sl = entry_price - atr * 1.5
            tp = entry_price + atr * 3.0
        else:
            sl = entry_price + atr * 1.5
            tp = entry_price - atr * 3.0
        return round(sl, 5), round(tp, 5)

    def decide_direction(self, score: float) -> Optional[str]:
        if score >= 0.5:
            return "BUY"
        elif score <= -0.5:
            return "SELL"
        else:
            return None


# === TELEGRAM BOT ===
class TelegramBot:
    def __init__(self, token: str, chat_id: int, bot_logic):
        self.token = token
        self.chat_id = chat_id
        self.bot_logic = bot_logic
        self.session = aiohttp.ClientSession()
        self.last_update_id = None

    async def send_message(self, text: str):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text}
        async with self.session.post(url, json=data) as resp:
            if resp.status != 200:
                logger.warning(f"Telegram send_message failed: {await resp.text()}")

    async def get_updates(self):
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"offset": self.last_update_id + 1 if self.last_update_id else None, "timeout": 20}
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"Telegram getUpdates failed: {await resp.text()}")
                return []
            data = await resp.json()
            return data.get("result", [])

    async def handle_commands(self):
        updates = await self.get_updates()
        for update in updates:
            self.last_update_id = update["update_id"]
            if "message" not in update:
                continue
            msg = update["message"]
            chat_id = msg["chat"]["id"]
            if chat_id != self.chat_id:
                await self.send_unauthorized(chat_id)
                continue
            if "text" not in msg:
                continue
            text = msg["text"].strip().lower()
            if text.startswith("/daily"):
                await self.cmd_daily()
            elif text.startswith("/weekly"):
                await self.cmd_weekly()
            elif text.startswith("/status"):
                await self.cmd_status()
            elif text.startswith("/maketrade"):
                await self.cmd_maketrade()
            elif text.startswith("/open"):
                await self.cmd_open()
            elif text.startswith("/closed"):
                await self.cmd_closed()
            else:
                await self.send_message("Unknown command.")

    async def send_unauthorized(self, chat_id):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": chat_id, "text": "Unauthorized access."}
        async with self.session.post(url, json=data):
            pass

    async def cmd_daily(self):
        pl = self.bot_logic.db.sum_pl_today()
        trades = self.bot_logic.db.count_trades_today()
        roi = (pl / (self.bot_logic.account_balance or 1)) * 100
        eod_roi = roi * (MAX_TRADES_DAY / max(trades, 1)) if trades > 0 else 0
        msg = (f"Daily P/L: £{pl:.2f}\n"
               f"Trades today: {trades}\n"
               f"ROI%: {roi:.2f}\n"
               f"Expected EOD ROI%: {eod_roi:.2f}")
        await self.send_message(msg)

    async def cmd_weekly(self):
        best = self.bot_logic.db.get_best_trade_week()
        total_pl = self.bot_logic.db.sum_pl_today() * 7  # rough weekly est.
        if best:
            msg = (f"Weekly P/L est: £{total_pl:.2f}\n"
                   f"Best trade: {best['pair']} P/L £{best['pl']:.2f} ROI {best['expected_roi']*100:.2f}%")
        else:
            msg = f"Weekly P/L est: £{total_pl:.2f}\nNo closed trades yet."
        await self.send_message(msg)

    async def cmd_status(self):
        status = (f"Balance: £{self.bot_logic.account_balance:.2f}\n"
                  f"Open trades: {len(self.bot_logic.open_trades)}\n"
                  f"Trades today: {self.bot_logic.db.count_trades_today()}\n"
                  f"Bot running: Yes")
        await self.send_message(status)

    async def cmd_maketrade(self):
        res = await self.bot_logic.evaluate_and_place_trade()
        await self.send_message(res)

    async def cmd_open(self):
        if not self.bot_logic.open_trades:
            await self.send_message("No open trades.")
            return
        msgs = []
        for t in self.bot_logic.open_trades:
            msgs.append(f"{t['pair']} {t['direction']} Units:{t['units']} Opened:{t['open_time']}")
        await self.send_message("\n".join(msgs))

    async def cmd_closed(self):
        trades = self.bot_logic.db.get_closed_trades()
        if not trades:
            await self.send_message("No closed trades.")
            return
        msgs = []
        for t in trades:
            msgs.append(f"{t['pair']} {t['direction']} P/L:£{t['pl']:.2f} ROI:{t['expected_roi']*100:.2f}% Duration:{t['duration_sec']//60}m")
        await self.send_message("\n".join(msgs))

    async def close(self):
        await self.session.close()


# === FOREX BOT ===
class ForexBot:
    def __init__(self):
        self.db = DB()
        self.oanda = OandaClient(OANDA_API_KEY, OANDA_ACCOUNT_ID)
        self.logic = TradeLogic()
        self.telegram = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, self)
        self.account_balance = 0.0
        self.open_trades: List[Dict[str, Any]] = []
        self.last_daily_reset = datetime.datetime.utcnow().date()

    async def refresh_account(self):
        summary = await self.oanda.get_account_summary()
        bal = float(summary["account"]["balance"])
        self.account_balance = bal
        logger.info(f"Account balance updated: £{bal:.2f}")

    async def refresh_open_trades(self):
        db_open = self.db.get_open_trades()
        oanda_positions = await self.oanda.get_open_positions()
        current_trade_ids = set(pos["tradeIDs"][0] if pos["tradeIDs"] else None for pos in oanda_positions.get("positions", []) if pos.get("tradeIDs"))
        # Remove trades from DB that are no longer open in OANDA
        for t in db_open:
            if t["trade_id"] not in current_trade_ids:
                self.db.update_trade_close(t["trade_id"], 0, datetime.datetime.utcnow().isoformat(), 0, 0, "CLOSED")
        self.open_trades = [t for t in db_open if t["status"] == "OPEN"]
        logger.info(f"Open trades refreshed: {len(self.open_trades)}")

    def in_peak_hours(self):
        now = datetime.datetime.utcnow()
        h = now.hour
        return any(start <= h < end for (start, end) in PEAK_HOURS_UTC)

    async def evaluate_and_place_trade(self) -> str:
        if len(self.open_trades) >= MAX_CONCURRENT:
            return "Max concurrent trades reached."

        trades_today = self.db.count_trades_today()
        if trades_today >= MAX_TRADES_DAY:
            return "Max trades per day reached."

        best_trade = None
        best_score = -math.inf
        for pair in PAIRS:
            try:
                candles_data = await self.oanda.get_instruments_candles(pair, count=50)
                candles = candles_data["candles"]
                features = self.logic.prepare_features(candles)
                score = self.logic.score_trade(features)
                direction = self.logic.decide_direction(score)
                if direction is None:
                    continue
                if score < 0.5:
                    continue  # skip low confidence

                entry_price = float(candles[-1]["mid"]["c"])
                sl, tp = self.logic.calculate_sl_tp(candles, entry_price, direction)

                risk_amount = self.account_balance * RISK_PCT
                sl_distance = abs(entry_price - sl)
                if sl_distance == 0:
                    continue
                units = int((risk_amount / sl_distance) * LEVERAGE)
                if units == 0:
                    continue

                expected_roi = score  # proxy for now

                if expected_roi < 0.01:
                    continue  # skip below 1% ROI

                if not self.in_peak_hours() and (score < 1.0 or expected_roi < 0.10):
                    continue  # After hours strict filter

                if expected_roi > best_score:
                    best_score = expected_roi
                    best_trade = {
                        "pair": pair,
                        "direction": direction,
                        "units": units if direction == "BUY" else -units,
                        "sl": sl,
                        "tp": tp,
                        "confidence": score,
                        "expected_roi": expected_roi,
                        "entry_price": entry_price
                    }
            except Exception as e:
                logger.error(f"Error evaluating {pair}: {e}")

        if best_trade is None:
            return "No suitable trade found."

        if best_trade["confidence"] < 0.75 and trades_today < MIN_TRADES_DAY:
            return "Confidence below 0.75 and min trades not reached."

        try:
            order_resp = await self.oanda.create_order(
                best_trade["pair"], best_trade["units"], best_trade["sl"], best_trade["tp"]
            )
            order_id = order_resp.get("orderCreateTransaction", {}).get("id")
            if not order_id:
                return "Failed to create order."

            trade_record = {
                "trade_id": order_id,
                "pair": best_trade["pair"],
                "open_time": datetime.datetime.utcnow().isoformat(),
                "direction": best_trade["direction"],
                "units": abs(best_trade["units"]),
                "open_price": best_trade["entry_price"],
                "stop_loss": best_trade["sl"],
                "take_profit": best_trade["tp"],
                "confidence": best_trade["confidence"],
                "expected_roi": best_trade["expected_roi"],
                "status": "OPEN"
            }
            self.db.insert_trade(trade_record)
            self.open_trades.append(trade_record)
            msg = (f"Trade placed: {best_trade['pair']} {best_trade['direction']} Units:{abs(best_trade['units'])} "
                   f"Risk: £{self.account_balance*RISK_PCT:.2f} Expected ROI: {best_trade['expected_roi']*100:.2f}% "
                   f"SL: {best_trade['sl']} TP: {best_trade['tp']}")
            logger.info(msg)
            return msg
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return "Failed to place trade."

    async def monitor_trades(self):
        now = datetime.datetime.utcnow()
        to_close = []
        for trade in self.open_trades:
            open_time = datetime.datetime.fromisoformat(trade["open_time"])
            elapsed = (now - open_time).total_seconds()
            if elapsed > TRADE_LIMIT_SEC:
                try:
                    price_resp = await self.oanda.get_instruments_candles(trade["pair"], count=1, granularity="M1")
                    last_price = float(price_resp["candles"][-1]["mid"]["c"])
                    pl = 0.0
                    if trade["direction"] == "BUY":
                        pl = (last_price - trade["open_price"]) * trade["units"]
                    else:
                        pl = (trade["open_price"] - last_price) * abs(trade["units"])
                    if pl > 0:
                        to_close.append((trade["trade_id"], last_price, now))
                except Exception as e:
                    logger.error(f"Error checking trade {trade['trade_id']} price: {e}")
        for tid, price, close_time in to_close:
            try:
                await self.oanda.close_trade(tid)
                open_trade = next((t for t in self.open_trades if t["trade_id"] == tid), None)
                if open_trade:
                    duration_sec = int((close_time - datetime.datetime.fromisoformat(open_trade["open_time"])).total_seconds())
                    pl = (price - open_trade["open_price"]) * open_trade["units"] if open_trade["direction"] == "BUY" else (open_trade["open_price"] - price) * abs(open_trade["units"])
                    self.db.update_trade_close(tid, price, close_time.isoformat(), pl, duration_sec, "CLOSED")
                    self.open_trades = [t for t in self.open_trades if t["trade_id"] != tid]
                    logger.info(f"Auto-closed trade {tid} after 2h profitable. P/L: £{pl:.2f}")
            except Exception as e:
                logger.error(f"Failed to close trade {tid}: {e}")

    async def daily_reset_check(self):
        today = datetime.datetime.utcnow().date()
        if today != self.last_daily_reset:
            self.last_daily_reset = today
            logger.info("Daily reset executed.")

    async def run(self):
        await self.refresh_account()
        await self.refresh_open_trades()
        while True:
            try:
                await self.telegram.handle_commands()
                await self.refresh_account()
                await self.refresh_open_trades()
                await self.monitor_trades()
                await self.daily_reset_check()
                await asyncio.sleep(TRADE_CHECK_INTERVAL)
            except Exception:
                logger.error("Unhandled exception in main loop", exc_info=True)
                await asyncio.sleep(5)


async def main():
    bot = ForexBot()
    try:
        await bot.run()
    finally:
        await bot.oanda.close()
        await bot.telegram.close()


if __name__ == "__main__":
    asyncio.run(main())
