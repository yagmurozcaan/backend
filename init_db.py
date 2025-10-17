"""
Database Initialization Module for NEUROLOOK Project
Creates SQLite database with reports table for storing autism detection results.
Handles database schema setup and directory creation for data persistence.
"""

import sqlite3
import os

DB_FILE = "reports/reports.db"

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            isim TEXT,
            soyisim TEXT,
            dosya TEXT,
            probability REAL,
            final_prediction TEXT,
            armflapping INTEGER,
            headbanging INTEGER,
            spinning INTEGER,
            blink INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("✅ SQLite veritabanı oluşturuldu:", DB_FILE)

if __name__ == "__main__":
    init_db()
