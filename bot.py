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
                kelly_multiplier REAL,
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
    account = await get_account()
    return float(account['account']['balance'])

async def get_open():
    resp = await retry_request(oanda.request, trades.OpenTrades(accountID=OANDA_ACCOUNT_ID))
    return resp.get('trades', [])

async def fetch_candles(p, gran='M5', cnt=100):
    params = {"granularity": gran, "count": cnt, "price": "M"}
    data = await retry_request(oanda.request, instruments.InstrumentsCandles(instrument=p, params=params))
    df = pd.DataFrame([{
        'time': c['time'],
        'o': float(c['mid']['o']),
        'h': float(c['mid']['h']),
        'l': float(c['mid']['l']),
        'c': float(c['mid']['c'])
    } for c in data.get('candles', [])])
    return df

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
    return {level: high - diff * level / 100 for level in FIB_LEVELS}

def check_fib_confluence(price, fibs, tolerance=0.0005):
    return any(abs(price - level_price) <= tolerance for level_price in fibs.values())

def kelly_criterion(win_rate, win_loss_ratio):
    return (win_rate - (1 - win_rate) / win_loss_ratio) if win_loss_ratio != 0 else 0

def calc_kelly_multiplier(p):
    stats = win_loss_tracker[p]
    win_rate = stats['wins'] / (stats['wins'] + stats['losses'])
    if stats['avg_loss'] == 0:
        win_loss_ratio = 0
    else:
        win_loss_ratio = stats['avg_win'] / stats['avg_loss']
    kelly = kelly_criterion(win_rate, win_loss_ratio)
    kelly_fraction = max(0, min(kelly * KELLY_FRACTION, 1))  # clamp between 0 and 1
    return kelly_fraction

def in_active_session():
    now = datetime.utcnow().time()
    start = time(hour=SESSION_START_HOUR)
    end = time(hour=SESSION_END_HOUR)
    return start <= now <= end

async def place(p, units, tp=None, sl=None, kelly_mult=1, conf_score=0.0, trade_reason=""):
    order_data = {
        "units": str(units),
        "instrument": p,
        "type": "MARKET",
        "timeInForce": "FOK",
        "positionFill": "DEFAULT"
    }
    if tp:
        order_data["takeProfitOnFill"] = {"price": f"{tp:.5f}"}
    if sl:
        order_data["stopLossOnFill"] = {"price": f"{sl:.5f}"}

    req = orders.OrderCreate(OANDA_ACCOUNT_ID, {"order": order_data})
    resp = await retry_request(oanda.request, req)
    logger.info(f"{p} order response: {resp}")

    open_trades = await get_open()
    for tr in open_trades:
        if tr['instrument'] == p and int(tr['currentUnits']) == units:
            await save_trade(tr, kelly_mult, conf_score, session_name, trade_reason)
            break

async def tick():
    global last_trade_times, trade_counts

    if not in_active_session():
        return  # Outside trading hours

    for p in ALLOWED_PAIRS:
        now = datetime.utcnow()
        # Enforce cooldown per pair
        if (now - last_trade_times[p]).total_seconds() < COOLDOWN_SECONDS:
            continue

        # Limit trades per session
        if trade_counts[p] >= MAX_TRADES_PER_SESSION:
            continue

        try:
            df = add_indicators(await fetch_candles(p))
            last, prev = df.iloc[-1], df.iloc[-2]
            bal, atr = await get_balance(), last['atr']

            # Kelly multiplier for position sizing
            kelly_mult = calc_kelly_multiplier(p)
            base_size = (bal * TRADE_RISK_PERCENT) / atr / 100000
            size = int(base_size * kelly_mult)
            if size == 0:
                continue

            # Signal checks with MACD + EMA + RSI + Fibonacci confluence
            macd_cross_up = prev['macd'] < prev['sig'] and last['macd'] > last['sig']
            macd_cross_down = prev['macd'] > prev['sig'] and last['macd'] < last['sig']

            ema_trend_up = last['ema_fast'] > last['ema_slow']
            ema_trend_down = last['ema_fast'] < last['ema_slow']

            fib_levels_dict = fib_levels(df['h'].max(), df['l'].min())

            # Check price near any fib level within tolerance
            fib_confluent = check_fib_confluence(last['c'], fib_levels_dict)

            buy_signal = macd_cross_up and ema_trend_up and last['rsi'] < 70 and fib_confluent
            sell_signal = macd_cross_down and ema_trend_down and last['rsi'] > 30 and fib_confluent

            if buy_signal or sell_signal:
                units = size if buy_signal else -size
                price = last['c']
                tp = price + atr * 3 if buy_signal else price - atr * 3
                sl = price - atr * 1.5 if buy_signal else price + atr * 1.5
                reason = "MACD+EMA+RSI+Fib"
                await place(p, units, tp, sl, kelly_mult, 1.0, reason)

                last_trade_times[p] = now
                trade_counts[p] += 1

        except Exception as e:
            logger.error(f"Error in tick for {p}: {e}")

async def update_win_loss(trade_outcome, p):
    stats = win_loss_tracker[p]
    if trade_outcome > 0:
        stats['wins'] += 1
        stats['avg_win'] = (stats['avg_win'] * (stats['wins'] - 1) + trade_outcome) / stats['wins']
        stats['last_result'] = "win"
    else:
        stats['losses'] += 1
        stats['avg_loss'] = (stats['avg_loss'] * (stats['losses'] - 1) + abs(trade_outcome)) / stats['losses']
        stats['last_result'] = "loss"

async def close_all_trades():
    open_trades = await get_open()
    closed = []
    for trade in open_trades:
        trade_id = trade['id']
        instrument = trade['instrument']
        units = -int(trade['currentUnits'])

        close_order = orders.OrderCreate(
            OANDA_ACCOUNT_ID,
            data={
                "order": {
                    "units": str(units),
                    "instrument": instrument,
                    "type": "MARKET",
                    "positionFill": "REDUCE_ONLY",
                    "timeInForce": "FOK"
                }
            }
        )
        try:
            resp = await retry_request(oanda.request, close_order)
            logger.info(f"Closed trade {trade_id} for {instrument}: {resp}")
            pl = float(trade.get('unrealizedPL', 0))
            await update_win_loss(pl, instrument)
            closed.append({'instrument': instrument, 'pl': pl})
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
    return closed

async def main_loop():
    await init_db()
    while True:
        try:
            await tick()
        except Exception as e:
            logger.error(f"Tick error: {e}")
        await asyncio.sleep(60)  # tick every 60 seconds for aggressive trading

# -- Added dummy trade function --
async def place_dummy_trade():
    """
    Place a dummy trade for the smallest amount on a random allowed pair,
    bypassing all logic and conditions.
    """
    p = random.choice(ALLOWED_PAIRS)
    units = 1  # smallest unit, always buy 1 unit
    try:
        req = orders.OrderCreate(
            OANDA_ACCOUNT_ID,
            data={
                "order": {
                    "units": str(units),
                    "instrument": p,
                    "type": "MARKET",
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }
        )
        resp = await retry_request(oanda.request, req)
        logger.info(f"Dummy trade placed on {p}: {resp}")

        # Save trade info
        open_trades = await get_open()
        for tr in open_trades:
            if tr['instrument'] == p and int(tr['currentUnits']) == units:
                await save_trade(tr, 0, 0.0, "Dummy Trade", "Manual dummy trade")
                break
    except Exception as e:
        logger.error(f"Failed to place dummy trade: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main_loop())
