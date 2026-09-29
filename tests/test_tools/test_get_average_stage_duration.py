import pytest

from tools.get_average_stage_duration import get_average_stage_duration
from tools.query_order_status import query_order_status


def test_returns_plausible_average_for_common_stage(db_conn):
    result = get_average_stage_duration("weaving", conn=db_conn)
    assert result["stage"] == "weaving"
    assert result["sample_size"] > 0
    # weaving's seeded avg is ~4 days +/- 1 day jitter
    assert 1 <= result["avg_days"] <= 8


def test_rejects_unknown_stage(db_conn):
    with pytest.raises(ValueError):
        get_average_stage_duration("not_a_real_stage", conn=db_conn)


def test_rejects_bad_sample_limit(db_conn):
    with pytest.raises(ValueError):
        get_average_stage_duration("weaving", sample_limit=0, conn=db_conn)
    with pytest.raises(ValueError):
        get_average_stage_duration("weaving", sample_limit=5000, conn=db_conn)


def test_surfaces_anomaly_1_dyeing_average_is_far_below_stuck_order(db_conn, anomalies):
    """Anomaly 1: the stuck order's time-in-stage should dwarf the average."""
    avg = get_average_stage_duration("dyeing", conn=db_conn)
    stuck_order = query_order_status(anomalies["stuck_dyeing_order_id"], conn=db_conn)
    days_stuck = (stuck_order["expected_delivery_date"].__class__.today()
                  - stuck_order["stage_entered_date"]).days
    assert days_stuck > avg["avg_days"] * 3
