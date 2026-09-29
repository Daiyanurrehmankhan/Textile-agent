"""Look up an order's current status and stage progression."""
from ._common import db_cursor, validate_positive_int

_ORDER_COLS = ["order_id", "buyer", "material", "quantity", "current_stage",
               "stage_entered_date", "expected_delivery_date"]


def query_order_status(order_id, conn=None):
    """Fetch one order's current status plus its stage history.

    Params:
        order_id (int): positive order_id to look up.
        conn: optional psycopg2 connection to reuse (e.g. a test fixture
            pinned to an isolated schema); a new connection is opened and
            closed automatically if omitted.

    Returns:
        dict with keys: order_id, buyer, material, quantity, current_stage,
        stage_entered_date, expected_delivery_date, stage_history (list of
        up to 50 dicts: stage, entered_at, exited_at, oldest first) --
        or None if no order has that id.

    Raises:
        ValueError: order_id is not a positive integer.
        ToolQueryError: the query failed.
    """
    validate_positive_int("order_id", order_id)

    with db_cursor(conn) as cur:
        cur.execute(
            f"""SELECT {', '.join(_ORDER_COLS)} FROM orders WHERE order_id = %s""",
            (order_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        order = dict(zip(_ORDER_COLS, row, strict=True))

        cur.execute(
            """SELECT stage, entered_at, exited_at FROM stage_history
               WHERE order_id = %s ORDER BY entered_at LIMIT 50""",
            (order_id,),
        )
        order["stage_history"] = [
            {"stage": s, "entered_at": e, "exited_at": x} for s, e, x in cur.fetchall()
        ]
    return order
