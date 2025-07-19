#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
import asyncio
from datetime import datetime

from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

bot = Bot(token=TELEGRAM_TOKEN)

async def unauthorized_response(update: Update):
    await update.message.reply_text("Unauthorized access. This bot only accepts commands from authorized users.")

def check_auth(update: Update):
    return update.effective_chat.id == AUTHORIZED_CHAT_ID

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update):
        return await unauthorized_response(update)
    proc = await asyncio.create_subprocess_exec(sys.executable, "trade_executor.py", "status", stdout=asyncio.subprocess.PIPE)
    out, _ = await proc.communicate()
    await update.message.reply_text(out.decode().strip())

async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update):
        return await unauthorized_response(update)
    # Simplified: call trade_executor for now
  
