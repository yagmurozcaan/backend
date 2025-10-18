import sqlite3
import os

DB_FILE = "reports/reports.db"

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # --- Genel rapor tablosu ---
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

    # --- Segment outlier tablosu ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS segment_outliers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            segment_index INTEGER,
            probability REAL,
            armflapping INTEGER,
            headbanging INTEGER,
            spinning INTEGER,
            blink INTEGER,
            FOREIGN KEY(report_id) REFERENCES reports(id)
        )
    """)

    conn.commit()
    conn.close()
    print("✅ SQLite veritabanı oluşturuldu:", DB_FILE)

if __name__ == "__main__":
    init_db()
