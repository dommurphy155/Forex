import os
import asyncio
import logging
from datetime import datetime, timedelta, time
import aiosqlite
import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints import accounts, trades, orders, instruments
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from config import (
    OANDA_API_KEY,
    OANDA_ACCOUNT_ID,
    DB_PATH,
    ALLOWED_PAIRS,
    TRADE_RISK_PERCENT,
    KELLY_FRACTION,
    EMA_FAST_PERIOD,
    EMA_SLOW_PERIOD,
    FIB_LEVELS,
    SESSION_START_HOUR,
    SESSION_END_HOUR,
    COOLDOWN_SECONDS,
    MAX_TRADES_PER_SESSION,
)
from ta.trend import macd, macd_signal, ema_indicator
from ta.momentum import rsi
from ta.volatility import average_true_range
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

oanda = API(access_token=OANDA_API_KEY, environment="practice")

last_trade_times = {p: datetime.min for p in ALLOWED_PAIRS}
trade_counts = {p: 0 for p in ALLOWED_PAIRS}
win_loss_tracker = {p: {"wins": 1, "losses": 1, "avg_win": 0.002, "avg_loss": 0.002, "last_result": None} for p in ALLOWED_PAIRS}
session_name = "London/New York Overlap"

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                trade_id TEXT UNIQUE,
                instrument TEXT,
                units INTEGER,
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                date TEXT,
                kelly_multiplier REAL DEFAULT 1.0,
                trade_confidence_score REAL,
                session_name TEXT,
                trade_reason TEXT
            )
        """)
        await db.commit()

async def save_trade(tr, kelly_mult, conf_score, sess, reason):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR IGNORE INTO trades
            (trade_id, instrument, units, entry_price, take_profit, stop_loss, date, kelly_multiplier, trade_confidence_score, session_name, trade_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tr['id'],
            tr['instrument'],
            int(tr['currentUnits']),
            float(tr['price']),
            float(tr.get('takeProfitOnFill', {}).get('price', 0)),
            float(tr.get('stopLossOnFill', {}).get('price', 0)),
            datetime.utcnow().isoformat(),
            kelly_mult,
            conf_score,
            sess,
            reason
        ))
        await db.commit()

async def retry_request(func, *args, **kwargs):
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=4)
    ):
        with attempt:
            return await asyncio.to_thread(func, *args, **kwargs)

async def get_account():
    return await retry_request(oanda.request, accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID))

async def get_balance():
    return float((await get_account())['account']['balance'])

async def get_open():
    return (await retry_request(oanda.request, trades.OpenTrades(accountID=OANDA_ACCOUNT_ID))).get('trades', [])

async def fetch_candles(p, gran='M5', cnt=100):
    data = await retry_request(oanda.request, instruments.InstrumentsCandles(instrument=p, params={"granularity": gran, "count": cnt, "price": "M"}))
    return pd.DataFrame([{'time':c['time'],'o':float(c['mid']['o']),'h':float(c['mid']['h']),'l':float(c['mid']['l']),'c':float(c['mid']['c'])} for c in data.get('candles', [])])

def add_indicators(df):
    df['macd'] = macd(df['c'], window_slow=26, window_fast=12)
    df['sig'] = macd_signal(df['c'], window_slow=26, window_fast=12, window_sign=9)
    df['rsi'] = rsi(df['c'], window=14)
    df['atr'] = average_true_range(df['h'], df['l'], df['c'], window=14)
    df['ema_fast'] = ema_indicator(df['c'], window=EMA_FAST_PERIOD)
    df['ema_slow'] = ema_indicator(df['c'], window=EMA_SLOW_PERIOD)
    df.dropna(inplace=True)
    return df

def fib_levels(high, low):
    diff = high - low
    return {level: high - diff * level/100 for level in FIB_LEVELS}

def check_fib_confluence(price, fibs, tolerance=0.0005):
    return any(abs(price - lvl) <= tolerance for lvl in fibs.values())

def kelly_criterion(win_rate, win_loss_ratio):
    return (win_rate - (1 - win_rate) / win_loss_ratio) if win_loss_ratio != 0 else 0

def calc_kelly_multiplier(p):
    stats = win_loss_tracker[p]
    wr = stats['wins']/(stats['wins']+stats['losses'])
    wl = stats['avg_win']/stats['avg_loss'] if stats['avg_loss'] else 0
    k = kelly_criterion(wr, wl)
    return max(0, min(k * KELLY_FRACTION, 1))

def in_active_session():
    now = datetime.utcnow().time()
    return time(SESSION_START_HOUR) <= now <= time(SESSION_END_HOUR)

async def place(p, units, tp=None, sl=None, kelly_mult=1, conf_score=0.0, trade_reason=""):
    data = {"order": {"units": str(units), "instrument": p, "type": "MARKET", "timeInForce": "FOK", "positionFill": "DEFAULT"}}
    if tp: data['order']["takeProfitOnFill"] = {"price": f"{tp:.5f}"}
    if sl: data['order']["stopLossOnFill"] = {"price": f"{sl:.5f}"}
    resp = await retry_request(oanda.request, orders.OrderCreate(OANDA_ACCOUNT_ID, data))
    logger.info(f"{p} order response: {resp}")
    for tr in await get_open():
        if tr['instrument']==p and int(tr['currentUnits'])==units:
            await save_trade(tr, kelly_mult, conf_score, session_name, trade_reason)
            break

async def tick():
    global last_trade_times, trade_counts
    if not in_active_session(): return
    for p in ALLOWED_PAIRS:
        now = datetime.utcnow()
        if (now - last_trade_times[p]).total_seconds() < COOLDOWN_SECONDS: continue
        if trade_counts[p] >= MAX_TRADES_PER_SESSION: continue
        try:
            df = add_indicators(await fetch_candles(p))
            last,prev = df.iloc[-1], df.iloc[-2]
            bal,atr = await get_balance(), last['atr']
            kf = calc_kelly_multiplier(p)
            size=int((bal * TRADE_RISK_PERCENT)/atr/100000*kf)
            if size==0: continue
            fuzzy= fib_levels(df['h'].max(), df['l'].min())
            buy = prev['macd']<prev['sig']<last['macd'] and last['rsi']<70 and check_fib_confluence(last['c'],fuzzy)
            sell = prev['macd']>prev['sig']>last['macd'] and last['rsi']>30 and check_fib_confluence(last['c'],fuzzy)
            if buy or sell:
                units = size if buy else -size
                price= last['c']
                tp=price+atr*3 if buy else price-atr*3
                sl=price-atr*1.5 if buy else price+atr*1.5
                await place(p, units, tp, sl, kf, 1.0, "MACD+EMA+RSI+Fib")
                last_trade_times[p], trade_counts[p] = now, trade_counts[p]+1
        except Exception as e:
            logger.error(f"Error in tick for {p}: {e}")

async def update_win_loss(pl, p):
    stats=win_loss_tracker[p]
    if pl>0:
        stats['wins']+=1
        stats['avg_win']=(stats['avg_win']*(stats['wins']-1)+pl)/stats['wins']
    else:
        stats['losses']+=1
        stats['avg_loss']=(stats['avg_loss']*(stats['losses']-1)+abs(pl))/stats['losses']

async def close_all_trades():
    res=[]
    for tr in await get_open():
        units=-int(tr['currentUnits'])
        try:
            await retry_request(oanda.request, orders.OrderCreate(
                OANDA_ACCOUNT_ID, {"order": {"units":str(units),"instrument":tr['instrument'],"type":"MARKET","positionFill":"REDUCE_ONLY","timeInForce":"FOK"}}
            ))
            pl=float(tr.get('unrealizedPL',0))
            await update_win_loss(pl, tr['instrument'])
            res.append({'instrument':tr['instrument'],'pl':pl})
        except Exception as e:
            logger.error(f"Failed to close trade {tr['id']}: {e}")
    return res

async def place_dummy_trade():
    p=random.choice(ALLOWED_PAIRS)
    units=1
    resp=await retry_request(oanda.request, orders.OrderCreate(
        OANDA_ACCOUNT_ID, {"order":{"units":str(units),"instrument":p,"type":"MARKET","timeInForce":"FOK","positionFill":"DEFAULT"}}
    ))
    logger.info(f"Dummy trade placed on {p}: {resp}")
    for tr in await get_open():
        if tr['instrument']==p and int(tr['currentUnits'])==units:
            await save_trade(tr, 0, 0.0, "Dummy Trade", "Manual dummy trade")
            break

async def main_loop():
    await init_db()
    while True:
        try: await tick()
        except Exception as e: logger.error(f"Tick error: {e}")
        await asyncio.sleep(60)

if __name__=="__main__":
    asyncio.run(main_loop())
