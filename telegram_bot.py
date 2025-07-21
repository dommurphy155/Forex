import os
import asyncio
import logging
import nest_asyncio
import aiosqlite
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from oandapyV20 import API
from oandapyV20.endpoints import accounts, trades, orders
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

# Apply nest_asyncio for Jupyter/event loop environments
nest_asyncio.apply()

# --- Config from .env ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "10"))
DB_PATH = os.getenv("DB_PATH", "./trades.db")
TRADE_RISK_PERCENT = float(os.getenv("TRADE_RISK_PERCENT", "0.02"))
DAILY_LOSS_CAP = float(os.getenv("DAILY_LOSS_CAP", "100"))

if not TELEGRAM_TOKEN or not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
    raise RuntimeError("Missing required environment variables")

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Globals ---
oanda = API(access_token=OANDA_API_KEY, environment="practice")
last_trade_time = datetime.utcnow() - timedelta(minutes=COOLDOWN_PERIOD)

# --- DB Init ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                instrument TEXT,
                units INTEGER,
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                date TEXT
            )
        """)
        await db.commit()

# --- DB Save ---
async def save_trade(trade):
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute("""
                INSERT OR IGNORE INTO trades (trade_id, instrument, units, entry_price, take_profit, stop_loss, date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['id'],
                trade['instrument'],
                int(trade['currentUnits']),
                float(trade['price']),
                float(trade.get('takeProfit', 0)),
                float(trade.get('stopLoss', 0)),
                datetime.utcnow().isoformat()
            ))
            await db.commit()
        except Exception as e:
            logger.error(f"Failed saving trade: {e}")

# --- OANDA Helpers ---
async def retry_oanda_request(func, *args, **kwargs):
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=4)
    ):
        with attempt:
            return await asyncio.to_thread(func, *args, **kwargs)

async def get_account_summary():
    return await retry_oanda_request(oanda.request, accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID))

async def get_account_balance():
    summary = await get_account_summary()
    return float(summary['account']['balance'])

async def get_open_trades():
    data = await retry_oanda_request(oanda.request, trades.OpenTrades(accountID=OANDA_ACCOUNT_ID))
    return data.get('trades', [])

# --- Indicators (placeholders) ---
def calculate_macd_rsi(df):
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'])
    return df

# --- Order logic (placeholder) ---
async def analyze_market_and_prepare_order():
    return {
        "instrument": "EUR_USD",
        "units": 1000,
        "takeProfit": 1.1100,
        "stopLoss": 1.0900,
        "price": 1.1000
    }

async def execute_trade(order):
    data = {
        "order": {
            "units": str(order["units"]),
            "instrument": order["instrument"],
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    resp = await retry_oanda_request(oanda.request, orders.OrderCreate(OANDA_ACCOUNT_ID, data))
    logger.info(f"Trade executed: {resp}")
    if 'orderCreateTransaction' in resp:
        await save_trade(resp['orderCreateTransaction'])
    return resp

# --- Helper for calculating daily and weekly ROI/loss ---

async def calculate_profit_since(start_time: datetime):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT entry_price, take_profit, stop_loss, date FROM trades WHERE date >= ?", (start_time.isoformat(),)
        )
        rows = await cursor.fetchall()
    if not rows:
        return 0.0
    # Simple P/L calc placeholder: sum difference between TP and entry or SL and entry
    profit = 0.0
    for entry_price, tp, sl, date_str in rows:
        # crude: Assume trade closed at TP if tp>0 else sl if sl>0 else 0
        if tp and tp > 0:
            profit += (tp - entry_price) * 10000  # pip value crude example
        elif sl and sl > 0:
            profit += (sl - entry_price) * 10000
    return profit

# --- Telegram handlers ---
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        bal = await get_account_balance()
        await update.message.reply_text(f"✅ All systems are operating smoothly.\n💰 Account balance: ${bal:.2f}\n🔄 Bot running without issues.")
    except Exception as e:
        logger.error(f"Status command error: {e}")
        await update.message.reply_text(f"❌ System check failed!\n⚠️ Issue: {e}")

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        start_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        profit = await calculate_profit_since(start_time)
        expected_roi = (profit / await get_account_balance()) * 100 if profit else 0
        await update.message.reply_text(f"📅 Daily Summary:\n"
                                        f"💸 Profit/Loss: ${profit:.2f}\n"
                                        f"📈 Expected ROI by EOD: {expected_roi:.2f}%")
    except Exception as e:
        logger.error(f"Daily command error: {e}")
        await update.message.reply_text(f"❌ Failed to fetch daily summary: {e}")

async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        start_time = datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())  # start of week (Monday)
        profit = await calculate_profit_since(start_time)
        expected_roi = (profit / await get_account_balance()) * 100 if profit else 0
        await update.message.reply_text(f"📅 Weekly Summary:\n"
                                        f"💸 Profit/Loss: ${profit:.2f}\n"
                                        f"📈 Expected ROI by EOW: {expected_roi:.2f}%")
    except Exception as e:
        logger.error(f"Weekly command error: {e}")
        await update.message.reply_text(f"❌ Failed to fetch weekly summary: {e}")

async def open_trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        trades = await get_open_trades()
        bal = await get_account_balance()
        if not trades:
            await update.message.reply_text("❌ No open trades found.")
            return
        lines = []
        total_roi = 0.0
        for t in trades:
            pl = float(t['unrealizedPL'])
            roi = (pl / bal) * 100 if bal else 0
            total_roi += roi
            lines.append(f"💹 {t['instrument']}: P&L ${pl:.2f} (~{roi:.2f}%)")
        lines.append(f"\n🔢 Total potential ROI from open trades: {total_roi:.2f}%")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        logger.error(f"Open trades error: {e}")
        await update.message.reply_text(f"❌ Could not fetch open trades: {e}")

async def maketrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_trade_time
    now = datetime.utcnow()
    if now < last_trade_time + timedelta(minutes=COOLDOWN_PERIOD):
        await update.message.reply_text("⏳ Cooldown active. Please wait before placing another trade.")
        return
    try:
        order = await analyze_market_and_prepare_order()
        await execute_trade(order)
        await update.message.reply_text(f"✅ Trade executed for {order['instrument']} with {order['units']} units. 💹")
        last_trade_time = now
    except Exception as e:
        logger.error(f"Trade error: {e}")
        await update.message.reply_text(f"❌ Trade failed: {e}")

# --- App runner ---
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("daily", daily))
    app.add_handler(CommandHandler("weekly", weekly))
    app.add_handler(CommandHandler("open", open_trades_cmd))
    app.add_handler(CommandHandler("maketrade", maketrade))
    return app

async def main():
    await init_db()
    app = build_app()
    logger.info("Bot started")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
