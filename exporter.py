import csv
import os
from database import create_connection

EXPORT_PATH = os.getenv("EXPORT_PATH") or "./exports"

def export_scraped_users(filename="scraped_users.csv"):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, username, first_name, messaged, replied FROM scraped_users
    """)
    rows = cursor.fetchall()
    conn.close()

    os.makedirs(EXPORT_PATH, exist_ok=True)
    filepath = os.path.join(EXPORT_PATH, filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "username", "first_name", "messaged", "replied"])
        writer.writerows(rows)

def export_sent_messages(filename="sent_messages.csv"):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, message_text, sent_at FROM sent_messages
    """)
    rows = cursor.fetchall()
    conn.close()

    os.makedirs(EXPORT_PATH, exist_ok=True)
    filepath = os.path.join(EXPORT_PATH, filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "message_text", "sent_at"])
        writer.writerows(rows)

def export_errors(filename="errors.csv"):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT error, timestamp FROM errors ORDER BY timestamp DESC LIMIT 100
    """)
    rows = cursor.fetchall()
    conn.close()

    os.makedirs(EXPORT_PATH, exist_ok=True)
    filepath = os.path.join(EXPORT_PATH, filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["error", "timestamp"])
        writer.writerows(rows)
