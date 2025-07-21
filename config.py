import os

# OANDA API credentials
API_TOKEN = os.getenv('OANDA_API_TOKEN')
ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')

# Trading settings
DEMO_MODE = True
TRADE_RISK_PERCENT = 2
DAILY_LOSS_CAP = 100
COOLDOWN_PERIOD = 5

# Telegram bot settings
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# SQLite database path
DB_PATH = 'trades.db'
