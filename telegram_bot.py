import os
import logging
import random
import time
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    referral_link = os.getenv("REFERRAL_LINK")
    message = get_start_message().format(referral_link=referral_link)
    await update.message.reply_text(f"Hello {user_name},\n\n{message}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    referral_link = os.getenv("REFERRAL_LINK")
    message = get_reply_template().format(referral_link=referral_link)
    await update.message.reply_text(message)

def main() -> None:
    application = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == "__main__":
    main()
