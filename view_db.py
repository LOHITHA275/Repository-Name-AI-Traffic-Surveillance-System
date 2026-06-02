import sqlite3

conn = sqlite3.connect("traffic.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM detected_plates")

count = cursor.fetchone()[0]

print("Total Records:", count)

cursor.execute("SELECT * FROM detected_plates")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()