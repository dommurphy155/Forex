import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_PATH = os.getenv("DB_PATH")
EXPORT_PATH = os.getenv("EXPORT_PATH")
GROUP_LINKS = os.getenv("GROUP_LINKS")
CHAT_IDS = os.getenv("CHAT_IDS")
REFERRAL_LINK = os.getenv("REFERRAL_LINK")
