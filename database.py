import sqlite3
import os

def create_connection():
    conn = sqlite3.connect(os.getenv("DB_PATH"))
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_message(chat_id: int, message: str):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (chat_id, message) VALUES (?, ?)", (chat_id, message))
    conn.commit()
    conn.close()

def fetch_messages():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    create_table()
