"""Run schema.sql against the Neon database pointed to by DATABASE_URL."""
from pathlib import Path

from db import get_connection


def main():
    sql = Path(__file__).with_name("schema.sql").read_text()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Schema applied.")


if __name__ == "__main__":
    main()
