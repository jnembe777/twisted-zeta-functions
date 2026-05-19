"""
Data module for database operations.

Contains database utilities and schema management.
"""

import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "arithmetics.db"
SCHEMA_PATH = DATA_DIR / "schema.sql"


def get_connection():
    """Get a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def initialize_database():
    """Initialize the database with the schema."""
    with open(SCHEMA_PATH, 'r') as f:
        schema = f.read()

    conn = get_connection()
    conn.executescript(schema)
    conn.commit()
    conn.close()
