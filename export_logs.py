import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger()
handler = RotatingFileHandler('forex.log', maxBytes=10_000_000, backupCount=5)
fmt = '{"time":"%(asctime)s","lvl":"%(levelname)s","msg":%(message)s}'
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)
