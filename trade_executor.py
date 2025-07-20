#!/usr/bin/env python3
import os, sys, json, time, logging, sqlite3, requests
from datetime import datetime, timezone

OANDA_API_KEY=os.getenv("OANDA_API_KEY"); OANDA_ACCOUNT_ID=os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_URL="https://api-fxpractice.oanda.com/v3"
DB="trade_executor.db"; MAX_D=20; MAX_C=1; RP=0.02; LIMIT=7200

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

class TradeExecutor:
    def __init__(self):
        self.conn=sqlite3.connect(DB,check_same_thread=False); self.conn.row_factory=sqlite3.Row
        self.init_db(); self.h={"Authorization":f"Bearer {OANDA_API_KEY}","Content-Type":"application/json"}

    def init_db(self):
        c=self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS trades(id TEXT PRIMARY KEY,...status TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS daily_stats(day TEXT PRIMARY KEY,trades_count INTEGER,profit REAL)")
        self.conn.commit()

    def balance(self):
        for _ in range(3):
            r=requests.get(f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/summary",headers=self.h,timeout=5)
            if r.status_code==200: return float(r.json()['account']['balance'])
            logging.warning("bal fail")
            time.sleep(1)
        raise RuntimeError("balance fetch failed")

    def open_trades(self):
        r=requests.get(f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/openTrades",headers=self.h,timeout=5)
        if r.status_code==200: return r.json()['trades']
        raise RuntimeError("openTrades failed")

    def place(self,pair,units,sl,tp):
        d={"order":{"instrument":pair,"units":str(units),"type":"MARKET","positionFill":"DEFAULT",
                     "stopLossOnFill":{"price":f"{sl:.5f}"},"takeProfitOnFill":{"price":f"{tp:.5f}"}}}
        for _ in range(3):
            r=requests.post(f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/orders",headers=self.h,json=d,timeout=10)
            if r.status_code in (200,201): return r.json()
            logging.warning("order fail")
            time.sleep(1)
        raise RuntimeError("order failed")

    def close(self,tid):
        for _ in range(3):
            r=requests.put(f"{OANDA_API_URL}/accounts/{OANDA_ACCOUNT_ID}/trades/{tid}/close",headers=self.h,timeout=5)
            if r.status_code==200: return r.json()
            logging.warning("close fail")
            time.sleep(1)
        raise RuntimeError("close failed")

    def save_open(self,tid,p,units,price,conf,roi,sl,tp):
        c=self.conn.cursor(); now=datetime.utcnow().isoformat()
        c.execute("INSERT OR IGNORE INTO trades(id,pair,units,open_price,open_time,confidence,expected_roi,stop_loss,take_profit,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (tid,p,units,price,now,conf,roi,sl,tp,"OPEN"))
        self.conn.commit()

    def update_close(self,tid,price,pl):
        c=self.conn.cursor(); ct=datetime.utcnow().isoformat()
        ot=self.conn.cursor().execute("SELECT open_time FROM trades WHERE id=?", (tid,)).fetchone()[0]
        dur=(datetime.fromisoformat(ct)-datetime.fromisoformat(ot)).total_seconds()
        c.execute("UPDATE trades SET close_price=?,close_time=?,profit_loss=?,duration_sec=?,status=? WHERE id=?",
                  (price,ct,pl,int(dur),"CLOSED",tid))
        self.conn.commit()

    def daily_ok(self,profit):
        c=self.conn.cursor(); day=datetime.utcnow().strftime("%Y-%m-%d")
        row=c.execute("SELECT trades_count,profit FROM daily_stats WHERE day=?", (day,)).fetchone()
        if row: c.execute("UPDATE daily_stats SET trades_count=trades_count+1,profit=profit+? WHERE day=?", (profit,day))
        else: c.execute("INSERT INTO daily_stats(day,trades_count,profit) VALUES(?,?,?)",(day,1,profit))
        self.conn.commit()

    def can_trade(self):
        day=datetime.utcnow().strftime("%Y-%m-%d")
        trades=self.conn.cursor().execute("SELECT trades_count FROM daily_stats WHERE day=?", (day,)).fetchone()
        return (not trades or trades[0]<MAX_D) and len(self.open_trades())<MAX_C

    def exec_trade(self,signal):
        if not self.can_trade(): return
        bal=self.balance(); risk=bal*RP
        units=int((risk*33)/signal['price'])
        o=self.place(signal['pair'],units,signal['stop_loss'],signal['take_profit'])
        tid=o['orderFillTransaction']['tradeOpened']['tradeID']
        self.save_open(tid,signal['pair'],units,signal['price'],signal['confidence'],signal['expected_roi'],signal['stop_loss'],signal['take_profit'])
        logging.info(f"placed {tid}")
        return tid

    def check_close(self):
        for t in self.open_trades():
            tid=t['id']; ot=datetime.fromisoformat(t['openTime'].replace('Z','+00:00'))
            if (datetime.utcnow().replace(tzinfo=timezone.utc)-ot).total_seconds()>LIMIT and float(t.get('unrealizedPL',0))>0:
                self.close(tid); self.update_close(tid,(t['price']),float(t.get('unrealizedPL',0)))
                self.daily_ok(float(t.get('unrealizedPL',0))); logging.info(f"auto-closed {tid}")

    def status(self):
        stats=self.conn.cursor().execute("SELECT trades_count,profit FROM daily_stats WHERE day=?", (datetime.utcnow().strftime("%Y-%m-%d"),)).fetchone()
        return stats if stats else (0,0.0)

    def open_list(self):
        return self.conn.cursor().execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()

    def closed_list(self):
        return self.conn.cursor().execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY close_time DESC LIMIT 5").fetchall()

def main_loop():
    from trader_ai_module import generate_best_signal
    te=TradeExecutor()
    while True:
        try:
            te.check_close()
            sig=generate_best_signal()
            te.exec_trade(sig)
        except Exception as e:
            logging.error(e)
        time.sleep(15)

if __name__=="__main__":
    from trader_ai_module import generate_best_signal
    te=TradeExecutor()
    if len(sys.argv)>1:
        if sys.argv[1]=="status":
            c,p=te.status(); print(f"Trades:{c}, P/L:{p:.2f}")
        elif sys.argv[1]=="open":
            print([dict(x) for x in te.open_list()])
        elif sys.argv[1]=="closed":
            print([dict(x) for x in te.closed_list()])
        else:
            print("Unknown")
    else:
        main_loop()
