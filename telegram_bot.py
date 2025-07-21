import os
import asyncio
import logging
import nest_asyncio
import aiosqlite
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ta  # for MACD, RSI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from oandapyV20 import API
from oandapyV20.endpoints import accounts, trades
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

# Apply nest_asyncio to avoid event loop conflicts (Jupyter etc)
nest_asyncio.apply()

# --- CONFIG ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "10"))  # in minutes
DB_PATH = os.getenv("DB_PATH", "./trades.db")
TRADE_RISK_PERCENT = float(os.getenv("TRADE_RISK_PERCENT", "0.02"))
DAILY_LOSS_CAP = float(os.getenv("DAILY_LOSS_CAP", "100"))

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, OANDA_API_KEY, OANDA_ACCOUNT_ID]):
    raise RuntimeError("Missing one or more required environment variables")

# --- LOGGING ---
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- OANDA Client ---
oanda = API(access_token=OANDA_API_KEY, environment='practice')

# --- Globals ---
last_trade_time = datetime.utcnow() - timedelta(minutes=COOLDOWN_PERIOD)

# --- DB Helper ---
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
            logger.error(f"Failed saving trade to DB: {e}")

async def fetch_trades_df():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT * FROM trades") as cursor:
            rows = await cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
            df['date'] = pd.to_datetime(df['date'])
            return df

# --- Retry wrapper for OANDA API ---
async def retry_oanda_request(func, *args, **kwargs):
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=4)
    ):
        with attempt:
            return await asyncio.to_thread(func, *args, **kwargs)

# --- OANDA API wrappers ---

async def get_account_summary():
    return await retry_oanda_request(oanda.request, accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID))

async def get_open_trades():
    data = await retry_oanda_request(oanda.request, trades.OpenTrades(accountID=OANDA_ACCOUNT_ID))
    return data.get('trades', [])

async def get_account_balance():
    summary = await get_account_summary()
    return float(summary['account']['balance'])

# --- Technical indicators helpers ---

def calculate_macd_rsi(df):
    # Requires df with 'close' column indexed by datetime
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'])
    return df

def calculate_atr(df, period=14):
    return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)

# --- Trading strategy ---

async def analyze_market_and_prepare_order():
    # This function should implement your market data fetch + MACD, RSI + ATR calculation
    # For demo, just return a dummy order dictionary

    # Fetch candles (daily or hourly) here from OANDA API or your data source

    # Example placeholder:
    order = {
        "instrument": "EUR_USD",
        "units": 1000,
        "takeProfit": 1.1100,
        "stopLoss": 1.0900,
        "price": 1.1000,
    }
    return order

async def execute_trade(order):
    # Build order request for OANDA
    data = {
        "order": {
            "units": str(order["units"]),
            "instrument": order["instrument"],
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            # Add TP and SL as needed in "takeProfitOnFill" and "stopLossOnFill"
        }
    }
    # Place order with retry
    resp = await retry_oanda_request(oanda.request, orders.OrderCreate(OANDA_ACCOUNT_ID, data))
    logger.info(f"Trade executed: {resp}")
    # Save trade info to DB
    if 'orderCreateTransaction' in resp:
        await save_trade(resp['orderCreateTransaction'])
    return resp

# --- Telegram Handlers ---

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        bal = await get_account_balance()
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                       text=f"✅ Bot online. Account balance: ${bal:.2f}")
    except Exception as e:
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                       text=f"❌ Status check failed: {e}")
        logger.error(f"Status command error: {e}")

async def open_trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        trades = await get_open_trades()
        if not trades:
            await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="No open trades.")
            return
        msg_lines = []
        bal = await get_account_balance()
        for t in trades:
            unrealizedPL = float(t['unrealizedPL'])
            roi = (unrealizedPL / bal) * 100 if bal else 0
            msg_lines.append(f"{t['instrument']}: P&L ${unrealizedPL:.2f} (~{roi:.2f}%)")
        msg_lines.append(f"\nTotal ROI: {sum((float(t['unrealizedPL'])/bal)*100 for t in trades):.2f}%")
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="\n".join(msg_lines))
    except Exception as e:
        logger.error(f"Open trades command error: {e}")
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Error fetching open trades: {e}")

async def maketrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_trade_time
    if datetime.utcnow() < last_trade_time + timedelta(minutes=COOLDOWN_PERIOD):
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="Cooldown active, wait before next trade.")
        return
    try:
        order = await analyze_market_and_prepare_order()
        await execute_trade(order)
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Trade executed for {order['instrument']} {order['units']} units.")
        last_trade_time = datetime.utcnow()
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Trade failed: {e}")

# --- Application setup and main ---

def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("open", open_trades_cmd))
    app.add_handler(CommandHandler("maketrade", maketrade))
    return app

async def main():
    await init_db()
    app = build_app()
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
