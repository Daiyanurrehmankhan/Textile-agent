"""Tests for tools/_common.py's failure handling -- specifically the polish
pass fix where a *connection* failure (not just a query failure) needed to
be caught and wrapped, not propagate raw.
"""
import psycopg2
import pytest

from tools._common import ToolQueryError, db_cursor
from tools.query_order_status import query_order_status


def test_connection_failure_is_wrapped_not_leaked(monkeypatch):
    """A failure in get_connection() itself (bad credentials, network down,
    a Neon cold-start hiccup) must come out as ToolQueryError, not a raw
    psycopg2 exception -- and the message must not contain the connection
    string."""
    def _boom():
        raise psycopg2.OperationalError(
            "could not connect to server: FATAL: password authentication failed "
            "postgresql://someuser:supersecret@example.neon.tech/db"
        )

    monkeypatch.setattr("tools._common.get_connection", _boom)

    with pytest.raises(ToolQueryError) as exc_info:
        query_order_status(1)  # conn=None -- forces db_cursor to call get_connection()

    message = str(exc_info.value)
    assert "supersecret" not in message
    assert "postgresql://" not in message


def test_db_cursor_query_failure_still_wraps_cleanly(db_conn):
    """Regression check: normal query-failure wrapping (not connection
    failure) still works after restructuring db_cursor's try block."""
    with pytest.raises(ToolQueryError):
        with db_cursor(db_conn) as cur:
            cur.execute("SELECT * FROM this_table_does_not_exist")
