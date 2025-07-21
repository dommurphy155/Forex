import asyncio, logging
from bot import get_account, get_open
logger = logging.getLogger()

async def hc():
    try:
        a = await get_account()
        o = await get_open()
        return bool(a and o is not None)
    except Exception:
        logger.error("Health check failed", exc_info=True)
        return False

if __name__=="__main__":
    print("✅ Bot healthy" if asyncio.run(hc()) else "❌ Bot unhealthy")
