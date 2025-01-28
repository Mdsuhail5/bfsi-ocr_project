import sqlite3
from contextlib import closing
from pathlib import Path


def create_connection(db_path="database/bfsi_ocr.db"):
    """
    Create a connection to the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Connection object.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_database(db_path="database/bfsi_ocr.db"):
    """
    Initialize the database by creating necessary tables.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    with create_connection(db_path) as conn:
        with closing(conn.cursor()) as cursor:
            # Create table for bank statements
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bank_statements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    description TEXT,
                    debit REAL,
                    credit REAL,
                    balance REAL
                )
            ''')

            # Create table for invoices
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    description TEXT,
                    quantity INTEGER,
                    unit_price REAL,
                    total REAL
                )
            ''')

            # Create table for profit and loss statements
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profit_loss (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    description TEXT,
                    amount REAL
                )
            ''')

            conn.commit()


def save_to_db(table_name, data, db_path="database/bfsi_ocr.db"):
    """
    Save processed data into the specified SQLite table.

    Args:
        table_name (str): Name of the table to save data.
        data (list of dict): Processed data to be saved.
        db_path (str): Path to the SQLite database file.
    """
    if not data:
        print("No data to save.")
        return

    with create_connection(db_path) as conn:
        with closing(conn.cursor()) as cursor:
            # Dynamically build the query based on keys from the first row
            placeholders = ', '.join(['?'] * len(data[0]))
            columns = ', '.join(data[0].keys())
            query = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'

            # Insert data
            cursor.executemany(query, [tuple(row.values()) for row in data])
            conn.commit()
            print(f"Data successfully saved to {table_name}.")


def insert_data(db_path, table_name, data):
    """
    Insert data into a specified table.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to insert data into.
        data (list of tuple): Data to insert, where each tuple is a row.
    """
    with create_connection(db_path) as conn:
        with closing(conn.cursor()) as cursor:
            placeholders = ', '.join(['?'] * len(data[0]))
            query = f'INSERT INTO {table_name} VALUES (NULL, {placeholders})'
            cursor.executemany(query, data)
            conn.commit()


def fetch_data(table_name, db_path="database/bfsi_ocr.db"):
    """
    Fetch all data from the specified SQLite table.

    Args:
        table_name (str): Name of the table to fetch data.
        db_path (str): Path to the SQLite database file.

    Returns:
        list of tuple: Retrieved rows.
    """
    with create_connection(db_path) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(f'SELECT * FROM {table_name}')
            return cursor.fetchall()
