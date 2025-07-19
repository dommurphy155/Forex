#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
import requests

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"
DB_PATH = "trade_executor.db"
MAX_TRADES_PER_DAY = 20
MAX_CONCURRENT_TRADES = 1
RISK_PCT = 0.02
TRADE_TIME_LIMIT_SEC = 2 * 3600

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

class TradeExecutor:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.headers = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

    def _init_db(self):
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
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            day TEXT PRIMARY KEY,
            trades_count INTEGER,
            profit REAL
        )""")
        self.conn.commit()

    def _get_account_balance(self):
        url = f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/summary"
        for _ in range(3):
            try:
                r = requests.get(url, headers=self.headers, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    return float(data['account']['balance'])
                logging.warning(f"Account balance fetch failed status {r.status_code}")
            except Exception as e:
                logging.warning(f"Account balance fetch error: {e}")
            time.sleep(1)
        raise RuntimeError("Failed to fetch account balance")

    def _get_open_trades(self):
        url = f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/openTrades"
        r = requests.get(url, headers=self.headers)
        if r.status_code == 200:
            trades = r.json()['trades']
            return trades
        raise RuntimeError(f"Failed to get open trades {r.status_code}")

    def _place_trade(self, pair, units, sl, tp):
        url = f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/orders"
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
                r = requests.post(url, headers=self.headers, json=data, timeout=10)
                if r.status_code in (200, 201):
                    return r.json()
                logging.warning(f"Order failed status {r.status_code} response {r.text}")
            except Exception as e:
                logging.warning(f"Order exception: {e}")
            time.sleep(1)
        raise RuntimeError("Failed to place trade")

    def _close_trade(self, trade_id):
        url = f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_id}/close"
        for _ in range(3):
            try:
                r = requests.put(url, headers=self.headers, timeout=5)
                if r.status_code == 200:
                    return r.json()
                logging.warning(f"Close trade failed {trade_id} status {r.status_code}")
            except Exception as e:
                logging.warning(f"Close trade exception {trade_id}: {e}")
            time.sleep(1)
        raise RuntimeError(f"Failed to close trade {trade_id}")

    def _save_trade_open(self, trade_id, pair, units, price, confidence, roi, sl, tp):
        c = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute("""
            INSERT OR IGNORE INTO trades (id, pair, units, open_price, open_time, confidence, expected_roi, stop_loss, take_profit, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, pair, units, price, now, confidence, roi, sl, tp, "OPEN"))
        self.conn.commit()

    def _update_trade_close(self, trade_id, close_price, profit_loss):
        c = self.conn.cursor()
        close_time = datetime.utcnow().isoformat()
        duration = int((datetime.fromisoformat(close_time) - datetime.fromisoformat(self._get_trade_open_time(trade_id))).total_seconds())
        c.execute("""
            UPDATE trades SET close_price=?, close_time=?, profit_loss=?, duration_sec=?, status=?
            WHERE id=?
        """, (close_price, close_time, profit_loss, duration, "CLOSED", trade_id))
        self.conn.commit()

    def _get_trade_open_time(self, trade_id):
        c = self.conn.cursor()
        c.execute("SELECT open_time FROM trades WHERE id=?", (trade_id,))
        row = c.fetchone()
        if row:
            return row['open_time']
        raise RuntimeError("Trade open time missing")

    def can_place_trade(self):
        c = self.conn.cursor()
        day = datetime.utcnow().strftime("%Y-%m-%d")
        c.execute("SELECT trades_count FROM daily_stats WHERE day=?", (day,))
        row = c.fetchone()
        if row and row['trades_count'] >= MAX_TRADES_PER_DAY:
            return False
        open_trades = self._get_open_trades()
        if len(open_trades) >= MAX_CONCURRENT_TRADES:
            return False
        return True

    def record_daily_trade(self, profit):
        c = self.conn.cursor()
        day = datetime.utcnow().strftime("%Y-%m-%d")
        c.execute("SELECT trades_count, profit FROM daily_stats WHERE day=?", (day,))
        row = c.fetchone()
        if row:
            c.execute("UPDATE daily_stats SET trades_count=trades_count+1, profit=profit+? WHERE day=?", (profit, day))
        else:
            c.execute("INSERT INTO daily_stats(day, trades_count, profit) VALUES (?, 1, ?)", (day, profit))
        self.conn.commit()

    def place_trade(self, pair, confidence, expected_roi, price, sl, tp):
        if not self.can_place_trade():
            logging.info("Cannot place trade: limit reached or open trades")
            return None
        balance = self._get_account_balance()
        risk_amount = balance * RISK_PCT
        # Calculate units based on price and risk, leverage 33
        units = int((risk_amount * 33) / price)
        order = self._place_trade(pair, units, sl, tp)
        trade_id = order['orderFillTransaction']['tradeOpened']['tradeID']
        self._save_trade_open(trade_id, pair, units, price, confidence, expected_roi, sl, tp)
        logging.info(f"Placed trade {trade_id} {pair} units {units} SL {sl} TP {tp}")
        return trade_id

    def check_and_close_trades(self):
        open_trades = self._get_open_trades()
        for t in open_trades:
            trade_id = t['id']
            open_time_str = t['openTime']
            open_time = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
            duration = (datetime.utcnow().replace(tzinfo=timezone.utc) - open_time).total_seconds()
            current_price = (float(t['price']) + float(t['currentUnits'])) / 2  # Approx mid-price placeholder
            unrealized_pl = float(t.get('unrealizedPL', 0.0))
            if duration > TRADE_TIME_LIMIT_SEC and unrealized_pl > 0:
                self._close_trade(trade_id)
                self._update_trade_close(trade_id, current_price, unrealized_pl)
                self.record_daily_trade(unrealized_pl)
                logging.info(f"Auto-closed trade {trade_id} after {duration}s with profit {unrealized_pl}")

    def get_open_trades_db(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM trades WHERE status='OPEN'")
        return c.fetchall()

    def get_closed_trades_db(self, limit=5):
        c = self.conn.cursor()
        c.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY close_time DESC LIMIT ?", (limit,))
        return c.fetchall()

    def get_daily_stats(self):
        c = self.conn.cursor()
        day = datetime.utcnow().strftime("%Y-%m-%d")
        c.execute("SELECT trades_count, profit FROM daily_stats WHERE day=?", (day,))
        row = c.fetchone()
        if row:
            return {"trades": row['trades_count'], "profit": row['profit']}
        return {"trades": 0, "profit": 0.0}

def main():
    execer = TradeExecutor()
    # Continuous monitor loop
    while True:
        try:
            execer.check_and_close_trades()
        except Exception as e:
            logging.error(f"Error during trade check: {e}")
        time.sleep(15)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        execer = TradeExecutor()
        if cmd == "status":
            stats = execer.get_daily_stats()
            print(f"Trades today: {stats['trades']}, P/L: {stats['profit']:.2f}")
        elif cmd == "open":
            trades = execer.get_open_trades_db()
            for t in trades:
                print(dict(t))
        elif cmd == "closed":
            trades = execer.get_closed_trades_db()
            for t in trades:
                print(dict(t))
        else:
            print("Unknown command")
    else:
        main()
