import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def plot_performance(db_path='trades.db', output_path='performance.png'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df['pnl'] = (df['take_profit'] - df['price']) * df['units']

    df.set_index('date', inplace=True)
    pnl_cumsum = df['pnl'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(pnl_cumsum.index, pnl_cumsum.values, label='Cumulative P&L', color='green')
    plt.title('Trading Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative P&L (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Performance chart saved to {output_path}")

if __name__ == "__main__":
    plot_performance()
