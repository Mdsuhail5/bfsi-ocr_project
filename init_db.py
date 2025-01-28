import sqlite3

# Connect to the database
conn = sqlite3.connect("bfsi_ocr.db")

# Create a cursor
cursor = conn.cursor()

# Create a table for storing transactions
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    category TEXT,
    amount REAL,
    date TEXT,
    description TEXT
)
""")

# Create a table for users
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT,
    email TEXT
)
""")

# Commit and close the connection
conn.commit()
conn.close()
print("Database initialized!")
