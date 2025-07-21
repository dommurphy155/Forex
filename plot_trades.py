import asyncio
import aiosqlite
import pandas as pd
import matplotlib.pyplot as plt

DB="./trades.db"
async def get_df():
    async with aiosqlite.connect(DB) as db:
        cur = await db.execute("SELECT * FROM trades")
        rows = await cur.fetchall()
        df=pd.DataFrame(rows,columns=[d[0] for d in cur.description])
        return df

df=asyncio.run(get_df())
if df.empty: print("No trades"); exit()
df.date = pd.to_datetime(df.date)
plt.plot(df.date, df.entry_price, marker='o')
plt.title("Trade Entries"); plt.xlabel("Time"); plt.ylabel("Price")
plt.show()
