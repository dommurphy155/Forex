#!/usr/bin/env python3
import os
import time
import math
import logging
from datetime import datetime, timezone
import numpy as np
import requests

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Config
PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CAD', 'AUD_USD', 'NZD_USD', 'USD_CHF']
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"
MODEL_PATH_LGB = "model_lightgbm.txt"
MODEL_PATH_XGB = "model_xgboost.json"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

def utcnow():
    return datetime.now(timezone.utc)

def get_candles(pair, count=500, granularity='M5'):
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    url = f"{OANDA_API_URL}/instruments/{pair}/candles"
    params = {"count": count, "granularity": granularity, "price": "M"}
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()['candles']
            logging.warning(f"Failed candles {pair} status {r.status_code}")
        except Exception as e:
            logging.warning(f"Exception fetching candles {pair}: {e}")
        time.sleep(1)
    raise RuntimeError(f"Failed fetching candles for {pair}")

def candles_to_arrays(candles):
    opens = np.array([float(c['mid']['o']) for c in candles])
    highs = np.array([float(c['mid']['h']) for c in candles])
    lows = np.array([float(c['mid']['l']) for c in candles])
    closes = np.array([float(c['mid']['c']) for c in candles])
    volumes = np.array([c.get('volume', 0) for c in candles])
    return opens, highs, lows, closes, volumes

def calculate_atr(highs, lows, closes, period=14):
    trs = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            abs(highs[1:] - closes[:-1]),
            abs(lows[1:] - closes[:-1])
        )
    )
    return np.mean(trs[-period:])

def feature_engineering(opens, highs, lows, closes, volumes):
    returns = (closes[1:] - closes[:-1]) / closes[:-1]
    vol = np.std(returns[-14:])
    atr = calculate_atr(highs, lows, closes)
    momentum = closes[-1] - closes[-14]
    return np.array([returns[-1], vol, atr, momentum])

def load_model():
    if lgb and os.path.exists(MODEL_PATH_LGB):
        model = lgb.Booster(model_file=MODEL_PATH_LGB)
        return model, 'lgb'
    if xgb and os.path.exists(MODEL_PATH_XGB):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH_XGB)
        return model, 'xgb'
    logging.error("No model file found. Exiting.")
    raise RuntimeError("Model file missing")

def predict_roi(model, model_type, features):
    if model_type == 'lgb':
        pred = model.predict(features.reshape(1, -1))
        conf = float(pred[0])
        roi = conf * 0.1
        return conf, roi
    elif model_type == 'xgb':
        prob = model.predict_proba(features.reshape(1, -1))[0][1]
        roi = prob * 0.1
        return prob, roi
    else:
        raise RuntimeError("Invalid model type")

def get_mid_price(pair):
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    url = f"{OANDA_API_URL}/pricing"
    params = {"instruments": pair}
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
            if r.status_code == 200:
                p = r.json()['prices'][0]
                bid = float(p['bids'][0]['price'])
                ask = float(p['asks'][0]['price'])
                return (bid + ask) / 2
            logging.warning(f"Failed price {pair} status {r.status_code}")
        except Exception as e:
            logging.warning(f"Exception getting price {pair}: {e}")
        time.sleep(1)
    raise RuntimeError(f"Failed to fetch price for {pair}")

def is_peak_hours():
    # London/NY overlap: 13:00-17:00 UTC
    now = datetime.utcnow().time()
    return (now >= datetime.strptime("13:00", "%H:%M").time() and
            now <= datetime.strptime("17:00", "%H:%M").time())

def calc_sl_tp(price, atr, expected_roi, pair):
    sl_pips = atr * 1.5
    tp_pips = sl_pips * max(expected_roi / 0.02, 1) * 1.5
    price_factor = 0.01 if "JPY" in pair else 0.0001
    sl = price - sl_pips * price_factor
    tp = price + tp_pips * price_factor
    return round(sl, 5), round(tp, 5)

def generate_best_signal():
    model, model_type = load_model()
    best_signal = None
    for pair in PAIRS:
        try:
            candles = get_candles(pair)
            opens, highs, lows, closes, volumes = candles_to_arrays(candles)
            features = feature_engineering(opens, highs, lows, closes, volumes)
            conf, roi = predict_roi(model, model_type, features)
            if conf < 0.5 or roi < 0.01:
                continue
            price = get_mid_price(pair)
            atr = calculate_atr(highs, lows, closes)
            sl, tp = calc_sl_tp(price, atr, roi, pair)
            if not is_peak_hours() and (conf < 1.0 or roi < 0.10):
                continue
            if not best_signal or conf > best_signal['confidence']:
                best_signal = {
                    "pair": pair,
                    "confidence": conf,
                    "expected_roi": roi,
                    "price": price,
                    "stop_loss": sl,
                    "take_profit": tp
                }
        except Exception as e:
            logging.error(f"Error generating signal for {pair}: {e}")
    if not best_signal:
        raise RuntimeError("No suitable trade signal found")
    return best_signal

def main():
    signal = generate_best_signal()
    print(signal)

if __name__ == "__main__":
    main()
