import os
import logging
import random
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

REFERRAL_LINK = os.getenv("REFERRAL_LINK")

START_MESSAGES = [
    "Hey there! 👋 Looking to boost your trading game? Check out this exclusive offer: {referral_link}",
    "Hi! 🚀 Want to take your trading to the next level? Here's something special for you: {referral_link}",
    "Hello! 💼 Ready to enhance your trading experience? Don't miss this: {referral_link}",
    "Greetings! 📈 Looking for better trading opportunities? Here's a link just for you: {referral_link}",
    "Hey! 💹 Want to improve your trading strategy? Check this out: {referral_link}",
]

REPLY_TEMPLATES = [
    "Thanks for reaching out! 😊 Here's the link again: {referral_link}",
    "Appreciate your interest! 👍 Don't forget to use this link: {referral_link}",
    "Glad to connect! 🙌 Here's the referral link you asked for: {referral_link}",
    "Thanks for your message! 📩 Here's the link: {referral_link}",
    "Great to hear from you! 💬 Here's the referral link: {referral_link}",
]

# Extra commands implementation

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Here you would fetch real status info from your system (DB or state)
    # For example purposes, using static data:
    msg = (
        "🟢 System Status: All systems operational! ✅\n"
        "📈 Scraper: Running\n"
        "📩 DM Sender: Running\n"
        "📊 Messages sent today: 123\n"
        "⚠️ No errors detected"
    )
    await update.message.reply_text(msg)


async def whatyoudoing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "🤖 Bot is currently scraping groups, collecting users, "
        "and sending personalized messages with your referral link."
    )
    await update.message.reply_text(msg)


async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    messages_sent = 123
    earnings_per_message = 0.08  # example
    estimated_earnings = messages_sent * earnings_per_message
    msg = (
        f"📅 Daily Report:\n"
        f"Messages sent: {messages_sent}\n"
        f"Estimated earnings today: £{estimated_earnings:.2f}"
    )
    await update.message.reply_text(msg)


async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    best_day = "Thursday"
    reason = "Highest engagement and response rate"
    msg = (
        f"📊 Weekly Report:\n"
        f"Best performing day: {best_day}\n"
        f"Reason: {reason}"
    )
    await update.message.reply_text(msg)


# Additional commands to increase communication and efficiency

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "📖 Available commands:\n"
        "/status - Get current system status\n"
        "/whatyoudoing - What the bot is doing now\n"
        "/daily - Today's message & earnings report\n"
        "/weekly - Best day of the week report\n"
        "/help - This message"
    )
    await update.message.reply_text(msg)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🏓 Pong! Bot is alive and kicking.")


async def referral(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = f"🔗 Here is your referral link:\n{REFERRAL_LINK}"
    await update.message.reply_text(msg)


def get_start_message():
    return random.choice(START_MESSAGES).format(referral_link=REFERRAL_LINK)


def get_reply_template():
    return random.choice(REPLY_TEMPLATES).format(referral_link=REFERRAL_LINK)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name or "there"
    message = get_start_message()
    await update.message.reply_text(f"Hello {user_name},\n\n{message}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = get_reply_template()
    await update.message.reply_text(message)


def main() -> None:
    application = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("whatyoudoing", whatyoudoing))
    application.add_handler(CommandHandler("daily", daily))
    application.add_handler(CommandHandler("weekly", weekly))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("ping", ping))
    application.add_handler(CommandHandler("referral", referral))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()


if __name__ == "__main__":
    main()
