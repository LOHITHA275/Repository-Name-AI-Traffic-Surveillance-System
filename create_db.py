import sqlite3

conn = sqlite3.connect("traffic.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE detected_plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    image_name TEXT,
    detected_time TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully")