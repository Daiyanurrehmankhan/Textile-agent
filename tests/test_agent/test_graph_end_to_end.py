"""End-to-end test: the compiled graph, run against the seeded test schema,
for a known delayed order (anomaly 1: stuck in dyeing). Makes real Gemini
API calls, so it's skipped when no credentials are configured.
"""
import pytest

from agent.graph import run_investigation
from config.settings import settings

pytestmark = pytest.mark.skipif(
    not settings.google_api_key,
    reason="requires GOOGLE_API_KEY in .env for a live Gemini call",
)


def test_investigates_a_known_delayed_order(db_conn, anomalies):
    result = run_investigation(
        order_id=anomalies["stuck_dyeing_order_id"],
        conn=db_conn,
        max_tool_calls=6,
    )

    assert result["final_explanation"]
    assert isinstance(result["final_explanation"], str)
    # It had to actually investigate -- not just guess from the order_id alone.
    assert len(result["tool_call_history"]) >= 1
    assert all("tool" in entry for entry in result["tool_call_history"])
