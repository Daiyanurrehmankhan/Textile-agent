import pytest

from tools.find_similar_past_delays import find_similar_past_delays


def test_returns_only_completed_rows_at_or_above_threshold(db_conn):
    rows = find_similar_past_delays("weaving", min_days=1, conn=db_conn)
    assert len(rows) > 0
    for row in rows:
        assert row["exited_at"] is not None
        assert row["duration_days"] >= 1

    # sorted longest-first
    durations = [r["duration_days"] for r in rows]
    assert durations == sorted(durations, reverse=True)


def test_high_threshold_returns_nothing(db_conn):
    # Seeded stage durations average a few days; nothing completed should
    # reach 300 days.
    assert find_similar_past_delays("weaving", min_days=300, conn=db_conn) == []


def test_rejects_unknown_stage(db_conn):
    with pytest.raises(ValueError):
        find_similar_past_delays("not_a_real_stage", min_days=1, conn=db_conn)


def test_rejects_bad_min_days(db_conn):
    for bad in (0, -1, 400):
        with pytest.raises(ValueError):
            find_similar_past_delays("weaving", min_days=bad, conn=db_conn)


def test_does_not_include_the_still_stuck_anomaly_orders(db_conn, anomalies):
    """Anomalies 1 & 4 are still IN PROGRESS (exited_at IS NULL), so this
    tool -- which only looks at completed stays -- must not surface them."""
    rows = find_similar_past_delays("dyeing", min_days=1, conn=db_conn)
    assert all(r["order_id"] != anomalies["stuck_dyeing_order_id"] for r in rows)
