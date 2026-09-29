import pytest

from tools.check_recent_inventory_movements import check_recent_inventory_movements


def test_rejects_unknown_material_id(db_conn):
    with pytest.raises(ValueError):
        check_recent_inventory_movements("NOT-A-REAL-ID", conn=db_conn)


def test_rejects_bad_days_and_limit(db_conn, anomalies):
    material_id = anomalies["depleted_material_id"]
    with pytest.raises(ValueError):
        check_recent_inventory_movements(material_id, days=0, conn=db_conn)
    with pytest.raises(ValueError):
        check_recent_inventory_movements(material_id, days=400, conn=db_conn)
    with pytest.raises(ValueError):
        check_recent_inventory_movements(material_id, limit=0, conn=db_conn)


def test_surfaces_anomaly_5_heavy_recent_activity_on_a_depleted_material(db_conn, anomalies):
    """Anomaly 5: the material is nearly out of stock (12 units) yet orders
    still in early stages are actively drawing on it -- recent movement
    should be non-empty, which is the mismatch worth flagging."""
    rows = check_recent_inventory_movements(anomalies["depleted_material_id"], days=60, conn=db_conn)
    assert len(rows) > 0
    assert all({"order_id", "stage", "timestamp", "notes", "order_quantity"} <= r.keys() for r in rows)
