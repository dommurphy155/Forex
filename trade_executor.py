#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import sqlite3
import requests
from datetime import datetime, timezone

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"
DB = "trade_executor.db"
MAX_D = 20  # max daily trades
MAX_C = 1   # max concurrent open trades
RP = 0.02   # risk percent of balance per trade
LIMIT = 7200  # max trade duration in seconds (2 hours)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')


class TradeExecutor:
    def __init__(self):
        self.conn = sqlite3.connect(DB, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_db()
        self.h = {
            "Authorization": f"Bearer {OANDA_API_KEY}",
            "Content-Type": "application/json"
        }

    def init_db(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            pair TEXT,
            units INTEGER,
            open_price REAL,
            open_time TEXT,
            confidence REAL,
            expected_roi REAL,
            stop_loss REAL,
            take_profit REAL,
            close_price REAL,
            close_time TEXT,
            profit_loss REAL,
            duration_sec INTEGER,
            status TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            day TEXT PRIMARY KEY,
            trades_count INTEGER,
            profit REAL
        )
        """)
        self.conn.commit()

    def balance(self):
        for _ in range(3):
            try:
                r = requests.get(
                    f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/summary",
                    headers=self.h,
                    timeout=5
                )
                if r.status_code == 200:
                    return float(r.json()['account']['balance'])
                logging.warning(f"Balance fetch failed with status {r.status_code}")
            except Exception as e:
                logging.warning(f"Balance fetch exception: {e}")
            time.sleep(1)
        raise RuntimeError("Balance fetch failed after retries")

    def open_trades(self):
        try:
            r = requests.get(
                f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/openTrades",
                headers=self.h,
                timeout=5
            )
            if r.status_code == 200:
                return r.json()['trades']
            raise RuntimeError(f"Open trades fetch failed with status {r.status_code}")
        except Exception as e:
            raise RuntimeError(f"Open trades fetch exception: {e}")

    def place(self, pair, units, sl, tp):
        data = {
            "order": {
                "instrument": pair,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{sl:.5f}"},
                "takeProfitOnFill": {"price": f"{tp:.5f}"}
            }
        }
        for _ in range(3):
            try:
                r = requests.post(
                    f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/orders",
                    headers=self.h,
                    json=data,
                    timeout=10
                )
                if r.status_code in (200, 201):
                    return r.json()
                logging.warning(f"Order place failed with status {r.status_code}")
            except Exception as e:
                logging.warning(f"Order place exception: {e}")
            time.sleep(1)
        raise RuntimeError("Order placement failed after retries")

    def close(self, trade_id):
        for _ in range(3):
            try:
                r = requests.put(
                    f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_id}/close",
                    headers=self.h,
                    timeout=5
                )
                if r.status_code == 200:
                    return r.json()
                logging.warning(f"Trade close failed with status {r.status_code}")
            except Exception as e:
                logging.warning(f"Trade close exception: {e}")
            time.sleep(1)
        raise RuntimeError("Trade close failed after retries")

    def save_open(self, trade_id, pair, units, price, confidence, roi, sl, tp):
        c = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute(
            "INSERT OR IGNORE INTO trades(id, pair, units, open_price, open_time, confidence, expected_roi, stop_loss, take_profit, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trade_id, pair, units, price, now, confidence, roi, sl, tp, "OPEN")
        )
        self.conn.commit()

    def update_close(self, trade_id, price, pl):
        c = self.conn.cursor()
        close_time = datetime.utcnow().isoformat()
        open_time = c.execute("SELECT open_time FROM trades WHERE id=?", (trade_id,)).fetchone()[0]
        duration = (datetime.fromisoformat(close_time) - datetime.fromisoformat(open_time)).total_seconds()
        c.execute(
            "UPDATE trades SET close_price=?, close_time=?, profit_loss=?, duration_sec=?, status=? WHERE id=?",
            (price, close_time, pl, int(duration), "CLOSED", trade_id)
        )
        self.conn.commit()

    def daily_ok(self, profit):
        c = self.conn.cursor()
        day = datetime.utcnow().strftime("%Y-%m-%d")
        row = c.execute("SELECT trades_count, profit FROM daily_stats WHERE day=?", (day,)).fetchone()
        if row:
            c.execute("UPDATE daily_stats SET trades_count=trades_count+1, profit=profit+? WHERE day=?", (profit, day))
        else:
            c.execute("INSERT INTO daily_stats(day, trades_count, profit) VALUES (?, ?, ?)", (day, 1, profit))
        self.conn.commit()

    def can_trade(self):
        day = datetime.utcnow().strftime("%Y-%m-%d")
        trades = self.conn.cursor().execute("SELECT trades_count FROM daily_stats WHERE day=?", (day,)).fetchone()
        open_trades = len(self.open_trades())
        return (not trades or trades[0] < MAX_D) and (open_trades < MAX_C)

    def exec_trade(self, signal):
        if not self.can_trade():
            logging.info("Daily or concurrent trade limits reached, skipping trade execution")
            return None
        balance = self.balance()
        risk = balance * RP
        units = int((risk * 33) / signal['price'])
        order_response = self.place(signal['pair'], units, signal['stop_loss'], signal['take_profit'])
        trade_id = order_response['orderFillTransaction']['tradeOpened']['tradeID']
        self.save_open(trade_id, signal['pair'], units, signal['price'], signal['confidence'], signal['expected_roi'], signal['stop_loss'], signal['take_profit'])
        logging.info(f"Placed trade {trade_id} for {signal['pair']}")
        return trade_id

    def _parse_open_time(self, tstr):
        if '.' in tstr:
            base, frac = tstr.split('.')
            frac = frac.rstrip('Z')
            frac = frac[:6].ljust(6, '0')
            tstr = f"{base}.{frac}+00:00"
        else:
            tstr = tstr.rstrip('Z') + "+00:00"
        return datetime.fromisoformat(tstr)

    def check_close(self):
        for trade in self.open_trades():
            trade_id = trade['id']
            try:
                open_time = self._parse_open_time(trade['openTime'])
            except Exception as e:
                logging.error(f"Failed to parse openTime '{trade['openTime']}' for trade {trade_id}: {e}")
                continue
            elapsed_sec = (datetime.utcnow().replace(tzinfo=timezone.utc) - open_time).total_seconds()
            unrealized_pl = float(trade.get('unrealizedPL', 0))
            current_price = float(trade.get('price', 0))
            if elapsed_sec > LIMIT and unrealized_pl > 0:
                self.close(trade_id)
                self.update_close(trade_id, current_price, unrealized_pl)
                self.daily_ok(unrealized_pl)
                logging.info(f"Auto-closed trade {trade_id} after {elapsed_sec:.0f}s with profit {unrealized_pl}")

    def status(self):
        c = self.conn.cursor()
        stats = c.execute("SELECT trades_count, profit FROM daily_stats WHERE day=?", (datetime.utcnow().strftime("%Y-%m-%d"),)).fetchone()
        return stats if stats else (0, 0.0)

    def open_list(self):
        c = self.conn.cursor()
        return c.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()

    def closed_list(self):
        c = self.conn.cursor()
        return c.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY close_time DESC LIMIT 5").fetchall()


def main_loop():
    from trader_ai_module import generate_best_signal
    te = TradeExecutor()
    while True:
        try:
            te.check_close()
            signal = generate_best_signal()
            te.exec_trade(signal)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        time.sleep(15)


if __name__ == "__main__":
    from trader_ai_module import generate_best_signal
    te = TradeExecutor()
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "status":
            count, profit = te.status()
            print(f"Trades: {count}, P/L: {profit:.2f}")
        elif cmd == "open":
            print([dict(row) for row in te.open_list()])
        elif cmd == "closed":
            print([dict(row) for row in te.closed_list()])
        else:
            print("Unknown command")
    else:
        main_loop()
