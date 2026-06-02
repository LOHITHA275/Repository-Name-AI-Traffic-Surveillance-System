from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

@app.route("/")
def dashboard():

    conn = sqlite3.connect("traffic.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM detected_plates
    ORDER BY id DESC
    """)
    plates = cursor.fetchall()

    cursor.execute("""
    SELECT COUNT(*)
    FROM detected_plates
    """)
    total_plates = cursor.fetchone()[0]

    conn.close()

    return render_template(
        "index.html",
        plates=plates,
        total_plates=total_plates
    )

if __name__ == "__main__":
    app.run(debug=True)