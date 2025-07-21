import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "10"))
DB_PATH = os.getenv("DB_PATH", "./trades.db")
TRADE_RISK_PERCENT = float(os.getenv("TRADE_RISK_PERCENT", "0.02"))
DAILY_LOSS_CAP = float(os.getenv("DAILY_LOSS_CAP", "100"))

MAX_LEVERAGE = 20
ALLOWED_PAIRS = os.getenv(
    "ALLOWED_PAIRS",
    "EUR_USD,GBP_USD,USD_JPY,USD_CAD,AUD_USD,NZD_USD,USD_CHF"
).split(",")
