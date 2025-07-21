import os
import random
import time
import logging
from telegram import Bot
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

START_MESSAGES = [
    "Hey there! 👋 Looking to boost your trading game? Check out this exclusive offer: {referral_link}",
    "Hi! 🚀 Want to take your trading to the next level? Here's something special for you: {referral_link}",
    "Hello! 💼 Ready to enhance your trading experience? Don't miss this: {referral_link}",
    "Greetings! 📈 Looking for better trading opportunities? Here's a link just for you: {referral_link}",
    "Hey! 💹 Want to improve your trading strategy? Check this out: {referral_link}"
]

REPLY_TEMPLATES = [
    "Thanks for reaching out! 😊 Here's the link again: {referral_link}",
    "Appreciate your interest! 👍 Don't forget to use this link: {referral_link}",
    "Glad to connect! 🙌 Here's the referral link you asked for: {referral_link}",
    "Thanks for your message! 📩 Here's the link: {referral_link}",
    "Great to hear from you! 💬 Here's the referral link: {referral_link}"
]

def get_start_message():
    return random.choice(START_MESSAGES)

def get_reply_template():
    return random.choice(REPLY_TEMPLATES)

def send_message(bot: Bot, chat_id: int, message: str) -> None:
    try:
        bot.send_message(chat_id=chat_id, text=message)
        logger.info(f"Message sent to {chat_id}")
    except Exception as e:
        logger.error(f"Failed to send message to {chat_id}: {e}")

def main() -> None:
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    chat_ids = os.getenv("CHAT_IDS").split(",")
    referral_link = os.getenv("REFERRAL_LINK")
    for chat_id in chat_ids:
        message = get_start_message().format(referral_link=referral_link)
        send_message(bot, chat_id, message)
        time.sleep(random.randint(5, 10))

if __name__ == "__main__":
    main()
