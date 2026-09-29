"""Demo: run the agent end-to-end against a real seeded case in the dev
database (a delay or an inventory mismatch), and print the full reasoning
trace.

Requires GOOGLE_API_KEY (see .env.example) and a dev DB already seeded
via `python db/generate_sample_data.py`.

Usage:
  python scripts/run_agent_demo.py [delay] [order_id]
  python scripts/run_agent_demo.py mismatch [material_id]
(order_id / material_id default to whichever case looks most anomalous, so
you can just run the script with no args to see each path work.)

Every reasoning/tool/explanation step the agent takes is also logged (via the
stdlib `logging` module -- agent/nodes.py only ever calls a logger, this
script is what decides the logs go to the console AND to logs/agent.log) so
a run can be analyzed after the fact, not just read off the summary trace.
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),  # appended across runs
    ],
)

from agent.graph import run_investigation, run_mismatch_investigation
from db import get_connection


def _pick_a_stuck_order(conn):
    """Find the order that's been sitting in its current stage the longest --
    i.e. anomaly 1 or 4 from db/generate_sample_data.py -- without hardcoding an id."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT order_id FROM orders
               WHERE current_stage NOT IN ('packed', 'shipped')
               ORDER BY CURRENT_DATE - stage_entered_date DESC
               LIMIT 1"""
        )
        return cur.fetchone()[0]


def _pick_a_mismatch_material(conn):
    """Find the inventory batch with the least plausible on-hand quantity --
    i.e. anomaly 2 from db/generate_sample_data.py -- without hardcoding an id."""
    with conn.cursor() as cur:
        cur.execute("SELECT material_id FROM inventory ORDER BY quantity_available DESC LIMIT 1")
        return cur.fetchone()[0]


def _print_result(result):
    print("=== Reasoning trace ===")
    for entry in result["tool_call_history"]:
        status = "ERROR" if entry["is_error"] else "ok"
        print(f"  step {entry['step']}: {entry['tool']}({entry['args']}) -> [{status}] {entry['result'][:300]}")

    print(f"\nconcluded_naturally={result['concluded_naturally']}  tool_calls_used={result['tool_call_count']}")
    print("\n=== Final explanation ===")
    print(result["final_explanation"])


def main():
    args = sys.argv[1:]
    mode = "delay"
    if args and args[0] in ("delay", "mismatch"):
        mode = args.pop(0)

    conn = get_connection()
    if mode == "delay":
        order_id = int(args[0]) if args else _pick_a_stuck_order(conn)
        print(f"Investigating order_id={order_id} (delay)...\n")
        # max_tool_calls left unset -- run_investigation defaults to
        # settings.default_max_tool_calls_delay (see config/settings.py).
        result = run_investigation(order_id=order_id, conn=conn)
    else:
        material_id = args[0] if args else _pick_a_mismatch_material(conn)
        print(f"Investigating material_id={material_id} (inventory mismatch)...\n")
        # Defaults to settings.default_max_tool_calls_mismatch -- higher than
        # the delay path's cap (see config/settings.py for why).
        result = run_mismatch_investigation(material_id=material_id, conn=conn)
    conn.close()

    _print_result(result)


if __name__ == "__main__":
    main()
