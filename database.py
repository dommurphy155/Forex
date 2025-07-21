import sqlite3
import os
from datetime import datetime

def create_connection():
    conn = sqlite3.connect(os.getenv("DB_PATH"), check_same_thread=False)
    return conn

def create_tables():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            username TEXT,
            first_name TEXT,
            scraped_at DATETIME,
            messaged BOOLEAN DEFAULT 0,
            replied BOOLEAN DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message_text TEXT,
            sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            replied BOOLEAN DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_scraped_user(user_id, username, first_name):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO scraped_users (user_id, username, first_name, scraped_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, username, first_name, datetime.now()))
        conn.commit()
    finally:
        conn.close()

def fetch_fresh_users(limit=20):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, username, first_name FROM scraped_users 
        WHERE messaged=0 LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def mark_user_messaged(user_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE scraped_users SET messaged=1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

def mark_user_replied(user_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE scraped_users SET replied=1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

def insert_sent_message(user_id, message_text):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sent_messages (user_id, message_text, sent_at)
        VALUES (?, ?, ?)
    """, (user_id, message_text, datetime.now()))
    conn.commit()
    conn.close()

def fetch_errors():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM errors ORDER BY timestamp DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return rows

def log_error(error_msg):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO errors (error) VALUES (?)", (error_msg,))
    conn.commit()
    conn.close()

def insert_message(chat_id, message):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (chat_id, message) VALUES (?, ?)", (chat_id, message))
    conn.commit()
    conn.close()

def fetch_messages():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    create_tables()
