"""Verify the database connection by running SELECT 1.

Run from the backend directory:  python check_db.py
"""

import sys

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.database.connection import engine


def main() -> int:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1")).scalar_one()
        print("Database connection successful.")
        return 0
    except SQLAlchemyError as exc:
        print(f"Database connection failed: {exc}")
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
