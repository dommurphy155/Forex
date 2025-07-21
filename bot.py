import os
import time
import json
import sqlite3
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from oandapyV20 import API
from oandapyV20.endpoints import accounts, orders, trades, positions
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from telegram import Bot
from config import *

# Initialize OANDA API client
api = API(access_token=API_TOKEN)

# Initialize Telegram bot
bot = Bot(token=TELEGRAM_TOKEN)

# SQLite database setup
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT,
                    action TEXT,
                    units INTEGER,
                    price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    date TIMESTAMP)''')
conn.commit()

# Function to fetch market data
def get_market_data(pair, granularity='M1', count=100):
    params = {
        'granularity': granularity,
        'count': count
    }
    response = requests.get(f'https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles', params=params)
    data = response.json()
    df = pd.DataFrame([{
        'time': candle['time'],
        'open': float(candle['mid']['o']),
        'high': float(candle['mid']['h']),
        'low': float(candle['mid']['l']),
        'close': float(candle['mid']['c'])
    } for candle in data['candles']])
    df['time'] = pd.to_datetime(df['time'])
    return df

# Function to calculate indicators
def calculate_indicators(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['macd'] = MACD(df['close']).macd()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    return df

# Function to place an order
def place_order(pair, units, stop_loss, take_profit):
    data = {
        "order": {
            "units": units,
            "instrument": pair,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {
                "price": stop_loss
            },
            "takeProfitOnFill": {
                "price": take_profit
            }
        }
    }
    r = orders.OrderCreate(ACCOUNT_ID, data=data)
    response = api.request(r)
    return response

# Function to log trade
def log_trade(pair, action, units, price, stop_loss, take_profit):
    cursor.execute('''INSERT INTO trades (pair, action, units, price, stop_loss, take_profit, date)
                      VALUES (?, ?, ?, ?, ?, ?, ?)''',
                   (pair, action, units, price, stop_loss, take_profit, datetime.now()))
    conn.commit()

# Function to send Telegram message
def send_telegram_message(message):
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# Function to generate trade report
def generate_trade_report():
    cursor.execute('''SELECT pair, action, units, price, stop_loss, take_profit, date FROM trades''')
    trades = cursor.fetchall()
    df = pd.DataFrame(trades, columns=['Pair', 'Action', 'Units', 'Price', 'Stop Loss', 'Take Profit', 'Date'])
    return df

# Main trading loop
def main():
    while True:
        # Fetch market data
        pair = 'EUR_USD'
        df = get_market_data(pair)
        df = calculate_indicators(df)

        # Trading logic
        if df['macd'].iloc[-1] > 0 and df['rsi'].iloc[-1] < 30:
            units = 1000
            stop_loss = df['close'].iloc[-1] - 0.0020
            take_profit = df['close'].iloc[-1] + 0.0040
            response = place_order(pair, units, stop_loss, take_profit)
            price = df['close'].iloc[-1]
            log_trade(pair, 'BUY', units, price, stop_loss, take_profit)
            send_telegram_message(f"Trade placed: BUY {pair} at {price}, SL: {stop_loss}, TP: {take_profit}")
        elif df['macd'].iloc[-1] < 0 and df['rsi'].iloc[-1] > 70:
            units = -1000
            stop_loss = df['close'].iloc[-1] + 0.0020
            take_profit = df['close'].iloc[-1] - 0.0040
            response = place_order(pair, units, stop_loss, take_profit)
            price = df['close'].iloc[-1]
            log_trade(pair, 'SELL', units, price, stop_loss, take_profit)
            send_telegram_message(f"Trade placed: SELL {pair} at {price}, SL: {stop_loss}, TP: {take_profit}")

        # Wait before next iteration
        time.sleep(60)

if __name__ == '__main__':
    main()
