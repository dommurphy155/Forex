#!/usr/bin/env python3
import os, sys, asyncio, subprocess, logging
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

async def unauthorized_response(update: Update):
    await update.message.reply_text("Unauthorized access.")

def check_auth(update: Update):
    return update.effective_chat.id == AUTHORIZED_CHAT_ID

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cmd = update.message.text.strip()
    if not check_auth(update):
        return await unauthorized_response(update)
    if cmd == "/daily":
        p = await asyncio.create_subprocess_exec(sys.executable, "trade_executor.py", "status", stdout=asyncio.subprocess.PIPE)
        out,_ = await p.communicate(); await update.message.reply_text(out.decode())
    elif cmd == "/open":
        p = await asyncio.create_subprocess_exec(sys.executable, "trade_executor.py", "open", stdout=asyncio.subprocess.PIPE)
        out,_ = await p.communicate(); await update.message.reply_text(out.decode())
    elif cmd == "/closed":
        p = await asyncio.create_subprocess_exec(sys.executable, "trade_executor.py", "closed", stdout=asyncio.subprocess.PIPE)
        out,_ = await p.communicate(); await update.message.reply_text(out.decode())
    elif cmd == "/status":
        p = await asyncio.create_subprocess_exec(sys.executable, "trade_executor.py", "status", stdout=asyncio.subprocess.PIPE)
        out,_ = await p.communicate(); await update.message.reply_text(out.decode())
    elif cmd == "/maketrade":
        p = await asyncio.create_subprocess_exec(sys.executable, "trade_executor.py", stdout=asyncio.subprocess.PIPE)
        out,_ = await p.communicate(); await update.message.reply_text(out.decode())
    else:
        await update.message.reply_text("Unknown command.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler(["daily","open","closed","status","maketrade"], handle))
    app.run_polling()

if __name__ == "__main__":
    main()
