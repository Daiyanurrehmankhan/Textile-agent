import pytest

from tools.query_inventory import query_inventory


def test_returns_all_inventory_up_to_limit(db_conn):
    rows = query_inventory(conn=db_conn, limit=200)
    assert len(rows) >= 15
    assert {"material_id", "material_name", "batch_number", "quantity_available", "location"} <= rows[0].keys()


def test_filters_by_material_id(db_conn):
    all_rows = query_inventory(conn=db_conn, limit=200)
    one_id = all_rows[0]["material_id"]
    rows = query_inventory(material_id=one_id, conn=db_conn)
    assert len(rows) == 1
    assert rows[0]["material_id"] == one_id


def test_rejects_bad_limit(db_conn):
    with pytest.raises(ValueError):
        query_inventory(limit=0, conn=db_conn)
    with pytest.raises(ValueError):
        query_inventory(limit=201, conn=db_conn)


def test_surfaces_anomaly_2_implausible_quantity(db_conn, anomalies):
    rows = query_inventory(material_id=anomalies["mismatch_material_id"], conn=db_conn)
    assert rows[0]["quantity_available"] == pytest.approx(9999.0)


def test_surfaces_anomaly_5_nearly_depleted_batch(db_conn, anomalies):
    rows = query_inventory(material_id=anomalies["depleted_material_id"], conn=db_conn)
    assert rows[0]["quantity_available"] == pytest.approx(12.0)
