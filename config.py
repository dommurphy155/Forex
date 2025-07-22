import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_PERIOD", "10"))  # seconds cooldown per pair
DB_PATH = os.getenv("DB_PATH", "./trades.db")
TRADE_RISK_PERCENT = float(os.getenv("TRADE_RISK_PERCENT", "0.02"))  # 2% risk per trade

DAILY_LOSS_CAP = float(os.getenv("DAILY_LOSS_CAP", "100"))
MAX_LEVERAGE = 20

ALLOWED_PAIRS = os.getenv(
    "ALLOWED_PAIRS",
    "EUR_USD,GBP_USD,USD_JPY,USD_CAD,AUD_USD,NZD_USD,USD_CHF"
).split(",")

# Added for aggressive strategy tuning:
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.5"))  # Fractional Kelly to control aggressiveness

EMA_FAST_PERIOD = int(os.getenv("EMA_FAST_PERIOD", "12"))  # for trend confirmation
EMA_SLOW_PERIOD = int(os.getenv("EMA_SLOW_PERIOD", "26"))

FIB_LEVELS = [38.2, 50.0, 61.8]  # Fibonacci retracement levels for confluence

SESSION_START_HOUR = int(os.getenv("SESSION_START_HOUR", "7"))   # 07:00 UTC - London Open
SESSION_END_HOUR = int(os.getenv("SESSION_END_HOUR", "17"))      # 17:00 UTC - NY close

MAX_TRADES_PER_SESSION = int(os.getenv("MAX_TRADES_PER_SESSION", "20"))  # Limit trades per pair per session
