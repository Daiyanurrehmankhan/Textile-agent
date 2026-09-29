"""Compute the average time orders spend in a given production stage."""
from ._common import db_cursor, validate_positive_int, validate_stage


def get_average_stage_duration(stage, sample_limit=200, conn=None):
    """Average duration (in days) orders spend in `stage`, based on
    completed stage_history rows (exited_at IS NOT NULL).

    Params:
        stage (str): one of tools._common.STAGES.
        sample_limit (int): max completed rows to average over
            (default 200, capped at 1000).
        conn: optional connection to reuse.

    Returns:
        dict: {stage, avg_days (float, or None if no completed rows exist),
        sample_size (int)}

    Raises:
        ValueError: stage isn't a recognized stage name, or sample_limit
            is out of range.
        ToolQueryError: the query failed.
    """
    validate_stage(stage)
    validate_positive_int("sample_limit", sample_limit, max_value=1000)

    with db_cursor(conn) as cur:
        cur.execute(
            """SELECT EXTRACT(EPOCH FROM (exited_at - entered_at)) / 86400.0
               FROM stage_history
               WHERE stage = %s AND exited_at IS NOT NULL
               ORDER BY entered_at DESC
               LIMIT %s""",
            (stage, sample_limit),
        )
        durations = [r[0] for r in cur.fetchall()]

    avg_days = round(sum(durations) / len(durations), 2) if durations else None
    return {"stage": stage, "avg_days": avg_days, "sample_size": len(durations)}
