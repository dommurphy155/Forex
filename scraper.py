import os
import asyncio
import logging
from telethon import TelegramClient, errors
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.types import ChannelParticipantsSearch
from database import insert_scraped_user, log_error
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "userbot.session")
GROUP_LINKS = [g.strip() for g in os.getenv("GROUP_LINKS", "").split(",") if g.strip()]

class Scraper:
    def __init__(self):
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

    async def start(self):
        await self.client.start()
        logger.info("Scraper client started")

    async def join_groups(self):
        for group in GROUP_LINKS:
            try:
                await self.client(JoinChannelRequest(group))
                logger.info(f"Joined group: {group}")
                await asyncio.sleep(5)  # avoid flood wait
            except errors.UserAlreadyParticipantError:
                logger.info(f"Already a participant in group: {group}")
            except Exception as e:
                log_error(f"Failed to join {group}: {e}")
                logger.error(f"Failed to join {group}: {e}")

    async def scrape_group_members(self, group, limit=200):
        try:
            logger.info(f"Scraping members from {group}")
            async for user in self.client.iter_participants(group, limit=limit, search=""):
                if user.bot or user.deleted:
                    continue  # skip bots and deleted users
                user_id = user.id
                username = user.username or ""
                first_name = user.first_name or ""
                insert_scraped_user(user_id, username, first_name)
            logger.info(f"Finished scraping members from {group}")
        except Exception as e:
            log_error(f"Error scraping members from {group}: {e}")
            logger.error(f"Error scraping members from {group}: {e}")

    async def run(self):
        await self.start()
        await self.join_groups()
        for group in GROUP_LINKS:
            await self.scrape_group_members(group)
            await asyncio.sleep(10)  # spacing between groups to avoid flood limits
        await self.client.disconnect()
        logger.info("Scraper client stopped")

if __name__ == "__main__":
    async def main():
        scraper = Scraper()
        await scraper.run()

    asyncio.run(main())
