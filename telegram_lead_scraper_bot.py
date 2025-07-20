import os
import sys
import asyncio
import logging
import random
import sqlite3
import csv
from datetime import datetime, timedelta
from telethon import TelegramClient, errors
from telethon.tl.functions.messages import ImportChatInviteRequest, CheckChatInviteRequest
from telethon.tl.types import PeerChannel
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))
if not BOT_TOKEN or not ADMIN_CHAT_ID:
    print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables.")
    sys.exit(1)

NICHES = [
    "crypto",
    "nft",
    "airdrop",
    "real estate",
    "dropshipping",
    "affiliate marketing",
    "make money online",
    "forex",
    "ecommerce"
]

MAX_GROUPS_PER_DAY = 20
MAX_USERS_PER_GROUP = 100
MIN_DELAY = 120
MAX_DELAY = 300

DB_PATH = "leads.db"
EXPORT_DIR = "exports"
SESSION_NAME = "anon_session"

os.makedirs(EXPORT_DIR, exist_ok=True)

logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
error_logger = logging.getLogger("error_logger")
error_handler = logging.FileHandler("bot_error.log")
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            bio TEXT,
            phone TEXT,
            group_name TEXT,
            niche TEXT,
            scrape_date TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS group_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invite_link TEXT UNIQUE,
            niche TEXT,
            last_attempt TEXT,
            success INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_join_count (
            date TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def add_group_to_queue(invite_link, niche):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO group_queue(invite_link, niche) VALUES (?,?)", (invite_link, niche))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def update_group_attempt(invite_link, success):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    if success:
        c.execute("UPDATE group_queue SET success=1, last_attempt=? WHERE invite_link=?", (now, invite_link))
    else:
        c.execute("UPDATE group_queue SET fail_count=fail_count+1, last_attempt=? WHERE invite_link=?", (now, invite_link))
    conn.commit()
    conn.close()

def increment_daily_join_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.utcnow().date().isoformat()
    c.execute("SELECT count FROM daily_join_count WHERE date=?", (today,))
    row = c.fetchone()
    if row:
        c.execute("UPDATE daily_join_count SET count=count+1 WHERE date=?", (today,))
    else:
        c.execute("INSERT INTO daily_join_count(date, count) VALUES (?,1)", (today,))
    conn.commit()
    conn.close()

def get_daily_join_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.utcnow().date().isoformat()
    c.execute("SELECT count FROM daily_join_count WHERE date=?", (today,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def save_lead(user, group_name, niche):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    scrape_date = datetime.utcnow().date().isoformat()
    try:
        c.execute("""
            INSERT OR IGNORE INTO leads(user_id, username, first_name, last_name, bio, phone, group_name, niche, scrape_date)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            user.id,
            user.username if hasattr(user, 'username') else None,
            user.first_name if hasattr(user, 'first_name') else None,
            user.last_name if hasattr(user, 'last_name') else None,
            user.about if hasattr(user, 'about') else None,
            user.phone if hasattr(user, 'phone') else None,
            group_name,
            niche,
            scrape_date
        ))
        conn.commit()
    except Exception as e:
        error_logger.error(f"Error saving lead {user.id}: {e}")
    finally:
        conn.close()

def export_leads_csv(niche):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date_str = datetime.utcnow().date().isoformat()
    filename = f"{EXPORT_DIR}/{niche}_leads_{date_str}.csv"
    c.execute("""
        SELECT user_id, username, first_name, last_name, bio, phone, group_name, scrape_date
        FROM leads WHERE niche=? AND scrape_date=?
    """, (niche, date_str))
    rows = c.fetchall()
    conn.close()
    with open(filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id','username','first_name','last_name','bio','phone','group_name','scrape_date'])
        writer.writerows(rows)
    return filename

async def send_file_to_admin(bot, filename, caption):
    try:
        await bot.send_document(chat_id=ADMIN_CHAT_ID, document=open(filename, 'rb'), caption=caption)
    except Exception as e:
        error_logger.error(f"Failed to send file {filename}: {e}")

async def discover_groups(bot):
    # Very simple scraper of public tg group directories (example tgstat.com)
    # Because there's no official API, we use hardcoded URLs with keyword search.
    # This is rudimentary and should be improved over time.
    import aiohttp
    discovered = 0
    async with aiohttp.ClientSession() as session:
        for niche in NICHES:
            if discovered >= MAX_GROUPS_PER_DAY:
                break
            url = f"https://tgstat.com/channels/search?q={niche}&page=1"
            try:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    # Parse all links like https://t.me/joinchat/XXXX or t.me/XXXXX from page
                    # Simple regex approach:
                    import re
                    links = set(re.findall(r"(https?://t\.me/joinchat/[A-Za-z0-9_-]+)", text))
                    for link in links:
                        if discovered >= MAX_GROUPS_PER_DAY:
                            break
                        add_group_to_queue(link, niche)
                        discovered += 1
            except Exception as e:
                error_logger.error(f"Error discovering groups for niche {niche}: {e}")
    logging.info(f"Discovered {discovered} new groups.")

async def join_and_scrape_groups(client, bot):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT invite_link, niche FROM group_queue WHERE success=0 ORDER BY fail_count ASC, last_attempt ASC LIMIT ?", (MAX_GROUPS_PER_DAY,))
    groups = c.fetchall()
    conn.close()
    joined_today = get_daily_join_count()
    for invite_link, niche in groups:
        if joined_today >= MAX_GROUPS_PER_DAY:
            logging.info("Reached max groups joined for today.")
            break
        try:
            if not invite_link.startswith("https://t.me/joinchat/"):
                logging.info(f"Skipping non-invite link: {invite_link}")
                update_group_attempt(invite_link, False)
                continue
            hash_code = invite_link.split('/')[-1]
            try:
                await client(ImportChatInviteRequest(hash_code))
            except errors.UserAlreadyParticipantError:
                pass
            except errors.InviteHashExpiredError:
                update_group_attempt(invite_link, False)
                continue
            except errors.InviteHashInvalidError:
                update_group_attempt(invite_link, False)
                continue
            dialogs = await client.get_dialogs()
            group = None
            for d in dialogs:
                if isinstance(d.entity, PeerChannel) and d.entity.access_hash is not None and d.entity.title:
                    if invite_link.split('/')[-1].lower() in str(d.entity):
                        group = d
                        break
            if not group:
                group = dialogs[-1]  # fallback to last dialog
            entity = group.entity
            members = await client.get_participants(entity, limit=MAX_USERS_PER_GROUP)
            for user in members:
                if user.bot or user.deleted:
                    continue
                save_lead(user, entity.title, niche)
            update_group_attempt(invite_link, True)
            increment_daily_join_count()
            joined_today += 1
            logging.info(f"Scraped group {entity.title} ({invite_link}) with {len(members)} members.")
            delay = random.randint(MIN_DELAY, MAX_DELAY)
            await asyncio.sleep(delay)
        except Exception as e:
            error_logger.error(f"Error joining or scraping group {invite_link}: {e}")
            update_group_attempt(invite_link, False)

async def scheduled_export_and_send(bot):
    for niche in NICHES:
        filename = export_leads_csv(niche)
        if os.path.getsize(filename) > 0:
            caption = f"Leads export for niche: {niche} date: {datetime.utcnow().date().isoformat()}"
            await send_file_to_admin(bot, filename, caption)

async def startjob(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_CHAT_ID:
        return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /startjob <invite_link or niche>")
        return
    arg = context.args[0].strip()
    if arg.startswith("https://t.me/joinchat/"):
        # Add group link directly
        niche = None
        if len(context.args) > 1:
            niche = context.args[1]
        add_group_to_queue(arg, niche or "unknown")
        await update.message.reply_text(f"Added group {arg} to queue.")
    else:
        # Add niche to niches (if new)
        if arg not in NICHES:
            NICHES.append(arg)
        await update.message.reply_text(f"Added niche '{arg}' to discovery list.")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_CHAT_ID:
        return
    count = get_daily_join_count()
    await update.message.reply_text(f"Groups joined today: {count}\nNiches tracked: {', '.join(NICHES)}")

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_CHAT_ID:
        return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /export <niche>")
        return
    niche = context.args[0].strip()
    filename = export_leads_csv(niche)
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        caption = f"Leads export for niche: {niche} date: {datetime.utcnow().date().isoformat()}"
        await send_file_to_admin(context.bot, filename, caption)
        await update.message.reply_text("Export sent.")
    else:
        await update.message.reply_text("No leads found for that niche.")

async def main_loop():
    init_db()
    bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
    bot_app.add_handler(CommandHandler("startjob", startjob))
    bot_app.add_handler(CommandHandler("stats", stats))
    bot_app.add_handler(CommandHandler("export", export))
    await bot_app.start()
    await bot_app.updater.start_polling()

    client = TelegramClient(SESSION_NAME, api_id=123456, api_hash="abcdef1234567890abcdef1234567890") # dummy placeholders, but we will run in bot-only mode
    # For bot-only mode with telethon, connect with bot token:
    await client.start(bot_token=BOT_TOKEN)

    bot = Bot(token=BOT_TOKEN)

    while True:
        try:
            await discover_groups(bot)
            await join_and_scrape_groups(client, bot)
            await scheduled_export_and_send(bot)
        except Exception as e:
            error_logger.error(f"Main loop error: {e}")
        await asyncio.sleep(3600 * 6)  # run every 6 hours

async def main():
    try:
        await main_loop()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")

if __name__ == "__main__":
    asyncio.run(main())
