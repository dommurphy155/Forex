import os
import time
import random
import logging
from telegram import Bot
from telegram.ext import ApplicationBuilder
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def join_group(bot: Bot, group_link: str) -> None:
    try:
        bot.join_chat(group_link)
        logger.info(f"Successfully joined group: {group_link}")
    except Exception as e:
        logger.error(f"Failed to join group {group_link}: {e}")

def main() -> None:
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    group_links = os.getenv("GROUP_LINKS").split(",")
    for group_link in group_links:
        join_group(bot, group_link)
        time.sleep(random.randint(5, 10))

if __name__ == "__main__":
    main()
