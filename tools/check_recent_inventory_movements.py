"""Recent order activity against a material, as a proxy for stock movement.

The schema has no dedicated stock-ledger table, so "movement" here means:
production_log entries for orders using this material within the last N
days -- each log entry marks work done (and presumably stock drawn) against
that order.
"""
from ._common import db_cursor, validate_positive_int

_COLS = ["order_id", "stage", "timestamp", "notes", "order_quantity"]


def check_recent_inventory_movements(material_id, days=30, limit=50, conn=None):
    """Recent production activity for orders using the material in
    `material_id`, within the last `days` days.

    Params:
        material_id (str): must match an existing inventory row.
        days (int): lookback window in days (default 30, capped at 365).
        limit (int): max rows returned (default 50, capped at 200).
        conn: optional connection to reuse.

    Returns:
        list[dict]: order_id, stage, timestamp, notes, order_quantity --
        most recent first.

    Raises:
        ValueError: material_id is empty/not a string, unknown, or
            days/limit are out of range.
        ToolQueryError: the query failed.
    """
    if not isinstance(material_id, str) or not material_id.strip():
        raise ValueError("material_id must be a non-empty string")
    validate_positive_int("days", days, max_value=365)
    validate_positive_int("limit", limit, max_value=200)

    with db_cursor(conn) as cur:
        cur.execute("SELECT material_name FROM inventory WHERE material_id = %s LIMIT 1", (material_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Unknown material_id {material_id!r}")
        material_name = row[0]

        cur.execute(
            """SELECT pl.order_id, pl.stage, pl.timestamp, pl.notes, o.quantity
               FROM production_logs pl
               JOIN orders o ON o.order_id = pl.order_id
               WHERE o.material = %s AND pl.timestamp >= NOW() - make_interval(days => %s)
               ORDER BY pl.timestamp DESC
               LIMIT %s""",
            (material_name, days, limit),
        )
        rows = cur.fetchall()
    return [dict(zip(_COLS, r, strict=True)) for r in rows]
