import sqlite3, os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    
    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    
    # Bots table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            owner_username TEXT
        )
    """)
    
    # Datasets table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER,
            filename TEXT,
            owner_username TEXT
        )
    """)
    
    # Annotations table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER,
            sentence TEXT,
            intent TEXT,
            entities TEXT
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize DB on import
if __name__ == "__main__":
    init_db()
    print("✅ Database initialized with tables: users, bots, datasets, annotations")
