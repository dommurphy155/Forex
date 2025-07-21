import os
import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from bot import get_balance, get_open, tick
from config import TELEGRAM_CHAT_ID
import nest_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
last_trade_time = datetime.utcnow() - timedelta(minutes=10)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bal = await get_balance()
    await update.message.reply_text(f"✅ Bot is online.\n💰 Current Balance: ${bal:.2f}")

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📅 Daily P&L report is not yet implemented. Stay tuned!")

async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📅 Weekly P&L report is coming soon!")

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ops = await get_open()
    if not ops:
        await update.message.reply_text("🛑 No open trades at the moment.")
    else:
        bal = await get_balance()
        msgs = [
            f"📈 {t['instrument']}: {float(t['unrealizedPL']):.2f} USD (~{100 * float(t['unrealizedPL']) / bal:.2f}%)"
            for t in ops
        ]
        total_roi = sum([100 * float(t['unrealizedPL']) / bal for t in ops])
        msgs.append(f"🔢 Total ROI: ~{total_roi:.2f}%")
        await update.message.reply_text("\n".join(msgs))

async def maketrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_trade_time
    now = datetime.utcnow()
    if now < last_trade_time + timedelta(minutes=10):
        await update.message.reply_text("⏳ Cooldown active. Please wait before sending another trade command.")
        return
    await update.message.reply_text("⚠️ Auto trade uses scheduled loop. Manual trade triggers are disabled.")
    last_trade_time = now

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 *Available Commands:*\n\n"
        "/status - Check bot status and balance\n"
        "/open - Show open trades and ROI\n"
        "/maketrade - Trigger auto trade (limited usage)\n"
        "/daily - Show daily P&L (coming soon)\n"
        "/weekly - Show weekly P&L (coming soon)\n"
        "/help - Show this help message\n"
    )
    await update.message.reply_markdown(help_text)

def build():
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_TOKEN")).build()
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("daily", daily))
    app.add_handler(CommandHandler("weekly", weekly))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("maketrade", maketrade))
    app.add_handler(CommandHandler("help", help_cmd))
    return app

async def main():
    nest_asyncio.apply()
    app = build()
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
