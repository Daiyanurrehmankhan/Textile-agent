"""End-to-end test: the compiled graph, run against the seeded test schema,
for a known inventory mismatch (anomaly 2: a batch whose recorded quantity
doesn't reconcile with order consumption). Makes real Gemini API calls, so
it's skipped when no credentials are configured.
"""
import pytest

from agent.graph import run_mismatch_investigation
from config.settings import settings

pytestmark = pytest.mark.skipif(
    not settings.google_api_key,
    reason="requires GOOGLE_API_KEY in .env for a live Gemini call",
)


def test_investigates_a_known_inventory_mismatch(db_conn, anomalies):
    result = run_mismatch_investigation(
        material_id=anomalies["mismatch_material_id"],
        conn=db_conn,
        max_tool_calls=6,
    )

    assert result["final_explanation"]
    assert isinstance(result["final_explanation"], str)
    # It had to actually investigate -- not just guess from the material_id alone.
    assert len(result["tool_call_history"]) >= 1
    # At least one of the two inventory-relevant tools should have been used.
    tools_called = {entry["tool"] for entry in result["tool_call_history"]}
    assert tools_called & {"query_inventory", "check_recent_inventory_movements"}
