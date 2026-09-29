"""Look up inventory batches, optionally filtered by material."""
from ._common import db_cursor, validate_positive_int

_INV_COLS = ["material_id", "material_name", "batch_number", "quantity_available", "location"]


def query_inventory(material_name=None, material_id=None, limit=50, conn=None):
    """Fetch inventory batches, optionally filtered by exact material match.

    Params:
        material_name (str|None): exact match on material_name, or None for any.
        material_id (str|None): exact match on material_id, or None for any.
        limit (int): max rows returned (default 50, capped at 200).
        conn: optional connection to reuse.

    Returns:
        list[dict]: material_id, material_name, batch_number,
        quantity_available, location -- ordered by material_id.

    Raises:
        ValueError: limit is out of range.
        ToolQueryError: the query failed.
    """
    validate_positive_int("limit", limit, max_value=200)

    where_clauses = []
    params = []
    if material_name is not None:
        where_clauses.append("material_name = %s")
        params.append(material_name)
    if material_id is not None:
        where_clauses.append("material_id = %s")
        params.append(material_id)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)

    with db_cursor(conn) as cur:
        cur.execute(
            f"""SELECT {', '.join(_INV_COLS)} FROM inventory {where_sql}
                ORDER BY material_id LIMIT %s""",
            params,
        )
        rows = cur.fetchall()
    return [dict(zip(_INV_COLS, r, strict=True)) for r in rows]
