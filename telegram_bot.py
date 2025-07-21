import os
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import httpx
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from config import (API_TOKEN, ACCOUNT_ID, DEMO_MODE, TRADE_RISK_PERCENT,
                    DAILY_LOSS_CAP, COOLDOWN_PERIOD,
                    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DB_PATH)
from oandapyV20 import API
from oandapyV20.endpoints import accounts, orders, trades, pricing
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

# Logger setup
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# SQLite
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# OANDA Client
oanda = API(access_token=API_TOKEN, environment='practice' if DEMO_MODE else 'live')

# Telegram Bot
bot = Bot(token=TELEGRAM_TOKEN)

# Rate-limit guard
last_trade_time = datetime.utcnow() - timedelta(minutes=COOLDOWN_PERIOD)

async def retry_request(func, *args, **kwargs):
    async for attempt in AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential()):
        with attempt:
            return await func(*args, **kwargs)

async def get_account_summary():
    return await retry_request(
        lambda: oanda.request(accounts.AccountSummary(ACCOUNT_ID))
    )

async def get_open_trades():
    resp = await retry_request(lambda: oanda.request(trades.OpenTrades(ACCOUNT_ID)))
    return resp.get('trades', [])

async def get_account_balance():
    data = await get_account_summary()
    bal = float(data['account']['balance'])
    return bal

# Health check endpoint
async def api_healthcheck():
    try:
        await get_account_summary()
        return True
    except Exception:
        return False

# Helper: load trades table
def df_trades():
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    return df

# Command handlers
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await api_healthcheck()
    health = "✅ All systems operational" if ok else "❌ API connection issues"
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=health)

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = df_trades()
    today = pd.Timestamp.utcnow().normalize()
    df['date'] = pd.to_datetime(df['date'])
    df_today = df[df['date'] >= today]
    pnl = ((df_today['take_profit'] - df_today['entry_price']) * df_today['units']).sum()
    roi = pnl / (await get_account_balance()) * 100
    await context.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"📅 Today's P&L: ${pnl:.2f}\n📈 ROI (est): {roi:.2f}%"
    )

async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = df_trades()
    monday = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=datetime.utcnow().weekday())
    df['date'] = pd.to_datetime(df['date'])
    df_week = df[df['date'] >= monday]
    pnl = ((df_week['take_profit'] - df_week['entry_price']) * df_week['units']).sum()
    roi = pnl / (await get_account_balance()) * 100
    await context.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"📅 Week's P&L: ${pnl:.2f}\n📈 Expected ROI: {roi:.2f}%"
    )

async def open_trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trades = await get_open_trades()
    if not trades:
        return await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="📭 No open trades right now.")
    msgs = []
    total_roi = 0
    bal = await get_account_balance()
    for t in trades:
        entry = float(t['price'])
        units = float(t['currentUnits'])
        unreal = (float(t['unrealizedPL']))
        roi = unreal / bal * 100
        msgs.append(f"{t['instrument']}: P&L ${unreal:.2f} (~{roi:.2f}%)")
        total_roi += roi
    msg = "\n".join(msgs) + f"\n\n🌟 Total ROI: {total_roi:.2f}%"
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

async def maketrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_trade_time
    if datetime.utcnow() < last_trade_time + timedelta(minutes=COOLDOWN_PERIOD):
        return await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="⏳ Cooldown active, wait before next trade.")
    # Insert your MACD+RSI+ATR strategy logic here
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="✅ Trade executed (placeholder)")
    last_trade_time = datetime.utcnow()

# App initialization
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
    asyncio.run(main())
