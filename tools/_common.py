"""Shared helpers for the tools/ package: DB access, validation, and the
one exception type all tools raise on a DB failure.
"""
import logging
from contextlib import contextmanager

import psycopg2

from db import get_connection

logger = logging.getLogger(__name__)

STAGES = ("spinning", "weaving", "dyeing", "finishing", "QC", "packed", "shipped")

class ToolQueryError(Exception):
    """A tool's DB query failed. Wraps the underlying driver error without
    leaking raw DB error text (which can include query fragments, data, or --
    for a connection failure -- connection details) to whatever's reading the
    exception message (an LLM, a caller). The real error is logged in full
    server-side (this module's logger) for debugging; it never travels in
    the exception itself."""


def validate_positive_int(name, value, max_value=None):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")


def validate_stage(stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")


@contextmanager
def db_cursor(conn=None):
    """Yield a cursor for one query. Opens (and closes) its own connection
    when `conn` is None; reuses the given connection otherwise -- e.g. a
    test fixture pinned to an isolated schema -- without closing it.

    get_connection() itself is inside the try block (not called before it)
    so a *connection* failure -- bad credentials, network blip, a transient
    Neon cold-start hiccup -- is caught and wrapped exactly like a query
    failure, instead of propagating a raw, unwrapped psycopg2 exception past
    this function (and past tool_node's error handling, which only expects
    ToolQueryError/ValueError) and crashing the whole agent run.
    """
    owns_conn = conn is None
    active_conn = None
    try:
        active_conn = get_connection() if owns_conn else conn
        with active_conn.cursor() as cur:
            yield cur
        active_conn.commit()
    except psycopg2.Error:
        logger.exception("Database operation failed")  # full detail, local log only -- never re-raised
        if active_conn is not None:
            active_conn.rollback()
        raise ToolQueryError("Database query failed.") from None
    finally:
        if owns_conn and active_conn is not None:
            active_conn.close()
