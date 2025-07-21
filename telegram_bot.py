import os, asyncio, logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from bot import get_balance, get_open, tick
from config import TELEGRAM_CHAT_ID
import nest_asyncio

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()
last= datetime.utcnow()-timedelta(minutes=10)

async def status(u,c):
    bal = await get_balance()
    await u.message.reply_text(f"✅ Online. Balance: ${bal:.2f}")

async def daily(u,c):
    await u.message.reply_text("📅 Daily P&L not yet implemented")

async def weekly(u,c):
    await u.message.reply_text("📅 Weekly P&L not yet implemented")

async def open_cmd(u,c):
    ops = await get_open()
    if not ops: await u.message.reply_text("No open trades")
    else:
        bal=await get_balance()
        msgs=[f"{t['instrument']}: {float(t['unrealizedPL']):.2f} (~{100*float(t['unrealizedPL'])/bal:.2f}%)" for t in ops]
        msgs.append(f"Total ROI ~ {sum([100*float(t['unrealizedPL'])/bal for t in ops]):.2f}%")
        await u.message.reply_text("\n".join(msgs))

async def maketrade(u,c):
    global last
    now = datetime.utcnow()
    if now < last + timedelta(minutes=10):
        await u.message.reply_text("⏳ Cooldown active")
        return
    await u.message.reply_text("⚠️ Auto trade uses scheduled loop")
    last=now

def build():
    app=ApplicationBuilder().token(os.getenv("TELEGRAM_TOKEN")).build()
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("daily", daily))
    app.add_handler(CommandHandler("weekly", weekly))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("maketrade", maketrade))
    return app

async def main():
    nest_asyncio.apply()
    app=build()
    await app.run_polling()

if __name__=="__main__":
    asyncio.run(main())
