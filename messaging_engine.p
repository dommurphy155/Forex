import os
import random
import asyncio
import logging
from database import insert_sent_message, mark_user_messaged, log_error

from telethon import TelegramClient, errors
from telethon.tl.types import InputPeerUser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "userbot.session")
REFERRAL_LINK = os.getenv("REFERRAL_LINK")

START_MESSAGES = [
    "Hey there! 👋 Looking to boost your trading game? Check this out: {referral_link}",
    "Hi! 🚀 Want to take your trading to the next level? Here's something special: {referral_link}",
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

class MessagingEngine:
    def __init__(self):
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        self.referral_link = REFERRAL_LINK

    async def start(self):
        await self.client.start()
        logger.info("Messaging client started")

    async def send_message_to_user(self, user_id, username, first_name):
        try:
            entity = await self.client.get_entity(user_id)
            message = random.choice(START_MESSAGES).format(referral_link=self.referral_link)
            await self.client.send_message(entity, message)
            insert_sent_message(user_id, message)
            mark_user_messaged(user_id)
            logger.info(f"Sent message to {username or user_id}")
            # Random delay between messages to avoid spam detection
            await asyncio.sleep(random.uniform(20, 60))
            return True
        except errors.FloodWaitError as e:
            logger.error(f"Flood wait error: sleeping for {e.seconds} seconds")
            await asyncio.sleep(e.seconds + 5)
            return False
        except Exception as e:
            log_error(str(e))
            logger.error(f"Failed to send message to {user_id}: {e}")
            return False

    async def send_batch_messages(self, users, batch_size=10):
        await self.start()
        count = 0
        for user in users:
            user_id, username, first_name = user
            success = await self.send_message_to_user(user_id, username, first_name)
            if success:
                count += 1
            if count >= batch_size:
                logger.info(f"Batch limit reached: {batch_size} messages sent")
                break
        await self.client.disconnect()
        logger.info("Messaging client stopped")

if __name__ == "__main__":
    import asyncio
    from database import fetch_fresh_users

    async def main():
        engine = MessagingEngine()
        users = fetch_fresh_users(20)
        await engine.send_batch_messages(users)

    asyncio.run(main())
