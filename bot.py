import os, asyncio, logging
from datetime import datetime, timedelta
import aiosqlite, pandas as pd, numpy as np
from oandapyV20 import API
from oandapyV20.endpoints import accounts, trades, orders, instruments
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from config import OANDA_API_KEY, OANDA_ACCOUNT_ID, DB_PATH, ALLOWED_PAIRS, TRADE_RISK_PERCENT, MAX_LEVERAGE
from ta.trend import macd, macd_signal
from ta.momentum import rsi
from ta.volatility import average_true_range

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

oanda = API(access_token=OANDA_API_KEY, environment="practice")
last_trade = datetime.utcnow() - timedelta(minutes=10)

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY, trade_id TEXT UNIQUE, instrument TEXT,
            units INTEGER, entry_price REAL, take_profit REAL,
            stop_loss REAL, date TEXT)""")
        await db.commit()

async def save_trade(tr):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""INSERT OR IGNORE INTO trades
            (trade_id,instrument,units,entry_price,take_profit,stop_loss,date)
            VALUES (?,?,?,?,?,?,?)""",
            (tr['id'], tr['instrument'], int(tr['currentUnits']),
             float(tr['price']), float(tr.get('takeProfit', 0)),
             float(tr.get('stopLoss', 0)), datetime.utcnow().isoformat()))
        await db.commit()

async def retry_request(func, *a, **k):
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=4)):
        with attempt:
            return await asyncio.to_thread(func, *a, **k)

async def get_account():
    return await retry_request(oanda.request, accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID))

async def get_balance():
    return float((await get_account())['account']['balance'])

async def get_open():
    return (await retry_request(oanda.request, trades.OpenTrades(accountID=OANDA_ACCOUNT_ID))).get('trades', [])

async def fetch_candles(p, gran='M5', cnt=100):
    params = {"granularity": gran, "count": cnt, "price": "M"}
    data = await retry_request(oanda.request, instruments.InstrumentsCandles(instrument=p, params=params))
    df = pd.DataFrame([{
        'time': c['time'], 'o': float(c['mid']['o']),
        'h': float(c['mid']['h']), 'l': float(c['mid']['l']),
        'c': float(c['mid']['c'])} for c in data.get('candles', [])])
    return df

def add_indicators(df):
    df['macd'] = macd(df['c'], window_slow=26, window_fast=12)
    df['sig'] = macd_signal(df['c'], window_slow=26, window_fast=12, window_sign=9)
    df['rsi'] = rsi(df['c'], window=14)
    df['atr'] = average_true_range(df['h'], df['l'], df['c'], window=14)
    return df.dropna()

async def place(p, units, tp, sl):
    req = orders.OrderCreate(OANDA_ACCOUNT_ID, {"order": {
        "units": str(units), "instrument": p, "type": "MARKET", "timeInForce": "FOK",
        "positionFill": "DEFAULT", "takeProfitOnFill": {"price": str(tp)},
        "stopLossOnFill": {"price": str(sl)}}})
    resp = await retry_request(oanda.request, req)
    logger.info(f"{p} order -> {resp}")
    if 'orderCreateTransaction' in resp:
        await save_trade(resp['orderCreateTransaction'])

async def tick():
    global last_trade
    for p in ALLOWED_PAIRS:
        df = add_indicators(await fetch_candles(p))
        last, prev = df.iloc[-1], df.iloc[-2]
        bal, atr = await get_balance(), last['atr']
        size = int((bal * TRADE_RISK_PERCENT) / atr / 100000)
        if size == 0: continue

        buy = prev['macd'] < prev['sig'] and last['macd'] > last['sig'] and last['rsi'] < 70
        sell = prev['macd'] > prev['sig'] and last['macd'] < last['sig'] and last['rsi'] > 30

        if buy or sell:
            units = size if buy else -size
            price = last['c']
            tp = price + atr * 3 if buy else price - atr * 3
            sl = price - atr * 1.5 if buy else price + atr * 1.5
            await place(p, units, tp, sl)
            last_trade = datetime.utcnow()

async def main_loop():
    await init_db()
    while True:
        try:
            await tick()
        except Exception as e:
            logger.error(e)
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main_loop())
