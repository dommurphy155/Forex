#!/usr/bin/env python3
import os, time, logging, math, requests
from datetime import datetime, timezone
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None

PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CAD', 'AUD_USD', 'NZD_USD', 'USD_CHF']
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"
MODEL_PATH_LGB = "model_lightgbm.txt"
MODEL_PATH_XGB = "model_xgboost.json"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

def get_candles(pair, count=500, granularity='M5'):
    for _ in range(3):
        r = requests.get(f"{OANDA_API_URL}/instruments/{pair}/candles",
                         headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
                         params={"count": count, "granularity": granularity, "price": "M"},
                         timeout=10)
        if r.status_code == 200:
            return r.json()['candles']
        logging.warning(f"failed candles {pair}: {r.status_code}")
        time.sleep(1)
    raise RuntimeError(f"candles failed for {pair}")

def candles_to_arrays(c):
    o,h,l,cl = np.array([float(candle['mid'][k]) for candle in c for k in ['o','h','l','c']]).reshape(-1,4).T
    v = np.array([candle.get('volume',0) for candle in c])
    return o, h, l, cl, v

def calculate_atr(h, l, cl, period=14):
    tr = np.maximum.reduce([h[1:]-l[1:], abs(h[1:]-cl[:-1]), abs(l[1:]-cl[:-1])])
    return np.mean(tr[-period:])

def feature_engineering(o,h,l,cl,v):
    ret = (cl[1:]-cl[:-1])/cl[:-1]
    return np.array([ret[-1], np.std(ret[-14:]), calculate_atr(h,l,cl), cl[-1]-cl[-14]])

def load_model():
    if lgb and os.path.exists(MODEL_PATH_LGB):
        return lgb.Booster(model_file=MODEL_PATH_LGB), 'lgb'
    if xgb and os.path.exists(MODEL_PATH_XGB):
        m = xgb.XGBClassifier(); m.load_model(MODEL_PATH_XGB); return m, 'xgb'
    raise RuntimeError("Missing model file")

def predict_roi(m,t,X):
    if t=='lgb':
        p = m.predict(X.reshape(1,-1))[0]; return float(p), float(p*0.1)
    p = m.predict(xgb.DMatrix(X.reshape(1,-1)))[0]; return float(p), float(p*0.1)

def get_mid_price(pair):
    for _ in range(3):
        r = requests.get(f"{OANDA_API_URL}/pricing", params={"instruments":pair},
                         headers={"Authorization":f"Bearer {OANDA_API_KEY}"}, timeout=5)
        if r.status_code==200:
            j=r.json()['prices'][0]; return (float(j['bids'][0]['price'])+float(j['asks'][0]['price']))/2
        logging.warning(f"price fail {pair}: {r.status_code}")
        time.sleep(1)
    raise RuntimeError(f"price fetch failed {pair}")

def is_peak():
    t = datetime.utcnow().time()
    return t >= datetime.strptime("13:00","%H:%M").time() and t <= datetime.strptime("17:00","%H:%M").time()

def calc_sl_tp(price,atr,roi,pair):
    slp=atr*1.5; tpp=slp*max(roi/0.02,1)*1.5
    f = 0.01 if "JPY" in pair else 0.0001
    return round(price - slp*f,5), round(price + tpp*f,5)

def generate_best_signal():
    model,mt=load_model(); best=None
    for p in PAIRS:
        cdl = get_candles(p)
        o,h,l,cl,v = candles_to_arrays(cdl)
        X = feature_engineering(o,h,l,cl,v)
        conf,roi = predict_roi(model,mt,X)
        if conf<0.5 or roi<0.01: continue
        price = get_mid_price(p); atr = calculate_atr(h,l,cl)
        sl,tp = calc_sl_tp(price,atr,roi,p)
        if not is_peak() and (conf<1.0 or roi<0.10): continue
        if not best or conf>best['confidence']:
            best={"pair":p,"confidence":conf,"expected_roi":roi,"price":price,"stop_loss":sl,"take_profit":tp}
    if not best: raise RuntimeError("No suitable trade signal")
    return best

if __name__=="__main__":
    print(generate_best_signal())
