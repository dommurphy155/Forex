export_logs.py
```python
import sqlite3
import pandas as pd

def export_trades_to_csv(db_path='trades.db', csv_path='trades.csv'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    df.to_csv(csv_path, index=False)
    conn.close()
    print(f"Exported {len(df)} trades to {csv_path}")

if __name__ == "__main__":
    export_trades_to_csv()
