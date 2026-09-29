"""Sanity-check the generated sample data: table counts, the 5 seeded
anomalies (found by query, not hardcoded ids), and one random normal order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `from db import ...` works when run directly

from db import get_connection

TABLES = ["orders", "inventory", "production_logs", "stage_history"]


def print_table_counts(cur):
    print("=== Table counts ===")
    for table in TABLES:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]}")
    print()


def print_stuck_orders(cur):
    # Orders sitting in a non-terminal stage the longest, oldest first.
    print("=== Anomaly 1 & 4: orders stuck in their current stage ===")
    cur.execute(
        """SELECT order_id, buyer, material, current_stage, stage_entered_date,
                  CURRENT_DATE - stage_entered_date AS days_in_stage
           FROM orders
           WHERE current_stage NOT IN ('packed', 'shipped')
           ORDER BY days_in_stage DESC
           LIMIT 2"""
    )
    for order_id, buyer, material, stage, entered, days in cur.fetchall():
        print(f"  order_id={order_id} | {buyer} | {material} | stage={stage} | "
              f"entered={entered} | days_in_stage={days}")
    print()


def print_qc_failures(cur):
    print("=== Anomaly 3: order with repeated QC failures ===")
    cur.execute(
        """SELECT order_id, COUNT(*) AS qc_fail_count
           FROM production_logs
           WHERE notes ILIKE '%QC FAILED%'
           GROUP BY order_id
           ORDER BY qc_fail_count DESC
           LIMIT 1"""
    )
    order_id, count = cur.fetchone()
    print(f"  order_id={order_id} has {count} QC FAILED log entries:")
    cur.execute(
        "SELECT timestamp, notes FROM production_logs WHERE order_id = %s AND notes ILIKE '%%QC FAILED%%' ORDER BY timestamp",
        (order_id,),
    )
    for ts, notes in cur.fetchall():
        print(f"    [{ts}] {notes}")
    print()


def print_inventory_extremes(cur):
    print("=== Anomaly 2: inventory quantity implausibly high ===")
    cur.execute(
        "SELECT material_id, material_name, batch_number, quantity_available FROM inventory ORDER BY quantity_available DESC LIMIT 1"
    )
    material_id, name, batch, qty = cur.fetchone()
    print(f"  material_id={material_id} | {name} | batch={batch} | quantity_available={qty}")

    print("=== Anomaly 5: inventory batch nearly depleted ===")
    cur.execute(
        "SELECT material_id, material_name, batch_number, quantity_available FROM inventory ORDER BY quantity_available ASC LIMIT 1"
    )
    material_id, name, batch, qty = cur.fetchone()
    print(f"  material_id={material_id} | {name} | batch={batch} | quantity_available={qty}")
    print()


def print_normal_order(cur):
    print("=== One random normal order (for comparison) ===")
    cur.execute(
        """SELECT order_id, buyer, material, quantity, current_stage,
                  stage_entered_date, expected_delivery_date
           FROM orders
           WHERE current_stage NOT IN ('dyeing', 'weaving')
              OR (CURRENT_DATE - stage_entered_date) < 10
           ORDER BY RANDOM()
           LIMIT 1"""
    )
    order_id, buyer, material, qty, stage, entered, expected = cur.fetchone()
    print(f"  order_id={order_id} | {buyer} | {material} | qty={qty} | stage={stage} | "
          f"entered={entered} | expected_delivery={expected}")


def main():
    conn = get_connection()
    cur = conn.cursor()

    print_table_counts(cur)
    print_stuck_orders(cur)
    print_qc_failures(cur)
    print_inventory_extremes(cur)
    print_normal_order(cur)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
