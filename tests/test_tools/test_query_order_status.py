import pytest

from tools.query_order_status import query_order_status


def test_returns_order_with_stage_history(db_conn, anomalies):
    order = query_order_status(anomalies["qc_fail_order_id"], conn=db_conn)
    assert order is not None
    assert order["order_id"] == anomalies["qc_fail_order_id"]
    assert order["current_stage"] == "QC"
    assert len(order["stage_history"]) >= 1
    assert all({"stage", "entered_at", "exited_at"} <= s.keys() for s in order["stage_history"])


def test_unknown_order_returns_none(db_conn):
    assert query_order_status(999_999, conn=db_conn) is None


def test_rejects_malformed_order_id(db_conn):
    for bad in (0, -1, "abc", 1.5, True):
        with pytest.raises(ValueError):
            query_order_status(bad, conn=db_conn)


def test_surfaces_anomaly_1_stuck_in_dyeing(db_conn, anomalies):
    """Anomaly 1: order stuck in dyeing far past the ~3 day average."""
    order = query_order_status(anomalies["stuck_dyeing_order_id"], conn=db_conn)
    assert order["current_stage"] == "dyeing"
    days_stuck = (order["expected_delivery_date"].__class__.today() - order["stage_entered_date"]).days
    assert days_stuck >= 15  # seeded at ~22 days, well past the ~3 day average
