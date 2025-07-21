import os
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from oandapyV20 import API
from oandapyV20.endpoints import accounts, trades
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

# Config import (adjust as per your config.py)
from config import (
    API_TOKEN, ACCOUNT_ID, DEMO_MODE, TRADE_RISK_PERCENT,
    DAILY_LOSS_CAP, COOLDOWN_PERIOD,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DB_PATH
)

# Logging setup
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Database setup
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# OANDA Client
oanda = API(access_token=API_TOKEN, environment='practice' if DEMO_MODE else 'live')

# Telegram Bot instance (mostly for lower-level calls if needed)
bot = Bot(token=TELEGRAM_TOKEN)

# Rate-limit cooldown guard
last_trade_time = datetime.utcnow() - timedelta(minutes=COOLDOWN_PERIOD)

async def retry_request(func, *args, **kwargs):
    async for attempt in AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential()):
        with attempt:
            return await func(*args, **kwargs)

async def get_account_summary():
    # Wrap sync oanda.request with asyncio.to_thread to not block event loop
    return await retry_request(lambda: asyncio.to_thread(oanda.request, accounts.AccountSummary(ACCOUNT_ID)))

async def get_open_trades():
    resp = await retry_request(lambda: asyncio.to_thread(oanda.request, trades.OpenTrades(ACCOUNT_ID)))
    return resp.get('trades', [])

async def get_account_balance():
    data = await get_account_summary()
    bal = float(data['account']['balance'])
    return bal

async def api_healthcheck():
    try:
        await get_account_summary()
        return True
    except Exception as e:
        logger.error(f"API Healthcheck failed: {e}")
        return False

def df_trades():
    try:
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        return df
    except Exception as e:
        logger.error(f"Failed to load trades from DB: {e}")
        return pd.DataFrame()

# Command Handlers
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await api_healthcheck()
    health = "✅ All systems operational" if ok else "❌ API connection issues"
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=health)

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = df_trades()
    if df.empty:
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="No trade data available for today.")
        return
    today = pd.Timestamp.utcnow().normalize()
    df['date'] = pd.to_datetime(df['date'])
    df_today = df[df['date'] >= today]
    pnl = ((df_today['take_profit'] - df_today['entry_price']) * df_today['units']).sum()
    bal = await get_account_balance()
    roi = pnl / bal * 100 if bal > 0 else 0
    msg = f"📅 Today's P&L: ${pnl:.2f}\n📈 ROI (est): {roi:.2f}%"
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = df_trades()
    if df.empty:
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="No trade data available for this week.")
        return
    monday = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=datetime.utcnow().weekday())
    df['date'] = pd.to_datetime(df['date'])
    df_week = df[df['date'] >= monday]
    pnl = ((df_week['take_profit'] - df_week['entry_price']) * df_week['units']).sum()
    bal = await get_account_balance()
    roi = pnl / bal * 100 if bal > 0 else 0
    msg = f"📅 Week's P&L: ${pnl:.2f}\n📈 Expected ROI: {roi:.2f}%"
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

async def open_trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trades = await get_open_trades()
    if not trades:
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="📭 No open trades right now.")
        return
    msgs = []
    total_roi = 0
    bal = await get_account_balance()
    for t in trades:
        unreal = float(t.get('unrealizedPL', 0))
        roi = (unreal / bal * 100) if bal > 0 else 0
        msgs.append(f"{t['instrument']}: P&L ${unreal:.2f} (~{roi:.2f}%)")
        total_roi += roi
    msg = "\n".join(msgs) + f"\n\n🌟 Total ROI: {total_roi:.2f}%"
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

async def maketrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_trade_time
    if datetime.utcnow() < last_trade_time + timedelta(minutes=COOLDOWN_PERIOD):
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="⏳ Cooldown active, wait before next trade.")
        return

    # TODO: Implement your MACD + RSI + ATR trading logic here. This placeholder must be replaced.
    # Make sure to log trades in DB and handle exceptions properly.

    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="✅ Trade executed (placeholder)")
    last_trade_time = datetime.utcnow()

# App builder
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("daily", daily))
    app.add_handler(CommandHandler("weekly", weekly))
    app.add_handler(CommandHandler("open", open_trades_cmd))
    app.add_handler(CommandHandler("maketrade", maketrade))
    return app

async def main():
    app = build_app()
    await app.initialize()
    await app.start()
    logger.info("Bot started")
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "already running" in str(e):
            loop = asyncio.get_event_loop()
            loop.create_task(main())
            loop.run_forever()
        else:
            raise
