import os, asyncio, logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from bot import get_balance, get_open, tick
from config import TELEGRAM_CHAT_ID
import nest_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
last = datetime.utcnow() - timedelta(minutes=10)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bal = await get_balance()
    await update.message.reply_text(
        f"✅ *Bot Status*\nOnline and running.\n💰 Balance: `${bal:,.2f}`",
        parse_mode="Markdown"
    )

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📅 Daily P&L not yet implemented.")

async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Weekly P&L not yet implemented.")

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ops = await get_open()
    if not ops:
        await update.message.reply_text("📭 No open trades.")
    else:
        bal = await get_balance()
        msgs = [f"💼 {t['instrument']}: `{float(t['unrealizedPL']):.2f}` (~{100 * float(t['unrealizedPL']) / bal:.2f}%)"
                for t in ops]
        msgs.append(f"📊 *Total ROI:* `{sum([100 * float(t['unrealizedPL']) / bal for t in ops]):.2f}%`")
        await update.message.reply_text("\n".join(msgs), parse_mode="Markdown")

async def maketrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last
    now = datetime.utcnow()
    if now < last + timedelta(minutes=10):
        await update.message.reply_text("⏳ Trade cooldown active. Try again later.")
        return
    await update.message.reply_text("⚙️ Trade request acknowledged.\nAuto-trading operates on a 5-min cycle.")
    last = now

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🧾 *Command Menu*\n"
        "/status - Check bot and balance status\n"
        "/open - Show current open trades\n"
        "/maketrade - Trigger manual trade (cooldown enforced)\n"
        "/daily - View today's P&L (WIP)\n"
        "/weekly - View weekly P&L (WIP)\n"
        "/help - Show this menu\n"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

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
