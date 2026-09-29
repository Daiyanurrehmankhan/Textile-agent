"""Find past orders that spent an unusually long time in a given stage."""
from ._common import db_cursor, validate_positive_int, validate_stage

_COLS = ["order_id", "stage", "entered_at", "exited_at", "duration_days"]


def find_similar_past_delays(stage, min_days, limit=10, conn=None):
    """Find completed stage_history rows where an order spent at least
    `min_days` in `stage` -- i.e. past cases similar to a currently-stuck
    order, for "has this happened before?" comparisons.

    Params:
        stage (str): one of tools._common.STAGES.
        min_days (int|float): minimum days spent in the stage to qualify
            (must be > 0 and <= 365).
        limit (int): max rows returned (default 10, capped at 100).
        conn: optional connection to reuse.

    Returns:
        list[dict]: order_id, stage, entered_at, exited_at, duration_days,
        sorted by duration_days descending (longest delay first).

    Raises:
        ValueError: stage unrecognized, or min_days/limit out of range.
        ToolQueryError: the query failed.
    """
    validate_stage(stage)
    if not isinstance(min_days, (int, float)) or isinstance(min_days, bool) or not (0 < min_days <= 365):
        raise ValueError(f"min_days must be a number in (0, 365], got {min_days!r}")
    validate_positive_int("limit", limit, max_value=100)

    with db_cursor(conn) as cur:
        cur.execute(
            """SELECT order_id, stage, entered_at, exited_at,
                      EXTRACT(EPOCH FROM (exited_at - entered_at)) / 86400.0 AS duration_days
               FROM stage_history
               WHERE stage = %s AND exited_at IS NOT NULL
                 AND EXTRACT(EPOCH FROM (exited_at - entered_at)) / 86400.0 >= %s
               ORDER BY duration_days DESC
               LIMIT %s""",
            (stage, min_days, limit),
        )
        rows = cur.fetchall()
    return [dict(zip(_COLS, r, strict=True)) for r in rows]
