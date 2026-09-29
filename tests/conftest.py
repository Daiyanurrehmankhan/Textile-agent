"""Shared pytest fixtures: an isolated Postgres schema seeded with the same
sample data (and 5 anomalies) as dev, via db.generate_sample_data.populate,
so tool tests never touch dev data.
"""
from pathlib import Path

import psycopg2
import pytest

from config.settings import settings
from db.generate_sample_data import populate

TEST_SCHEMA = settings.test_schema
SCHEMA_SQL = (Path(__file__).resolve().parent.parent / "schema.sql").read_text()


@pytest.fixture(scope="session")
def _seeded():
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
        cur.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
        cur.execute(f"SET search_path TO {TEST_SCHEMA}")
        cur.execute(SCHEMA_SQL)

    anomalies = populate(conn)

    yield conn, anomalies

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
    conn.close()


@pytest.fixture(scope="session")
def db_conn(_seeded):
    """psycopg2 connection pinned to the isolated test schema, already seeded."""
    return _seeded[0]


@pytest.fixture(scope="session")
def anomalies(_seeded):
    """order_id/material_id of the 5 seeded anomalies -- see
    db/generate_sample_data.populate for what each one is."""
    return _seeded[1]
