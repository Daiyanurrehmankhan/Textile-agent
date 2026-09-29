"""Populate the DB with realistic fake textile mill data, including 5 seeded
anomalies for the reconciliation agent (and its tests) to find later.

`populate(conn)` does the actual work and is reused by tests/conftest.py to
seed an isolated test schema with the same anomalies -- single source of
truth for sample data generation. Idempotent: truncates the 4 tables first,
so re-running never duplicates data.
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `from db import ...` works when run directly

from faker import Faker

from db import get_connection

fake = Faker()
random.seed(42)  # reproducible sample data -- same anomalies land on the same ids every run
Faker.seed(42)

STAGES = ["spinning", "weaving", "dyeing", "finishing", "QC", "packed", "shipped"]
STAGE_AVG_DAYS = {"spinning": 3, "weaving": 4, "dyeing": 3, "finishing": 2, "QC": 1, "packed": 1, "shipped": 0}

MATERIALS = [
    "30s combed cotton yarn",
    "20s carded cotton yarn",
    "polyester blend fabric",
    "cotton-polyester twill",
    "viscose rayon yarn",
    "denim fabric 12oz",
]

NEAT_NOTES = {
    "spinning": "Spinning completed within spec, count verified.",
    "weaving": "Weaving completed, fabric width and GSM checked OK.",
    "dyeing": "Dye lot matched to standard, shade approved.",
    "finishing": "Finishing (calendering/coating) completed on schedule.",
    "QC": "QC passed, no defects noted.",
    "packed": "Packed and staged for dispatch.",
    "shipped": "Shipped, dispatch docs filed.",
}
MESSY_NOTES = [
    "shift 2 forgot to log start time, checked w/ supervisor after",
    "machine 4 acting up again, ran slower than usual",
    "op said fabric felt rough??? flagged for review",
    "power cut ~20 min, resumed after, no impact expected",
    "batch mixed up w/ another order briefly, sorted out",
]
QC_FAIL_NOTES = [
    "QC FAILED - color shade mismatch vs approved swatch, batch on hold",
    "QC FAILED - shrinkage above tolerance, rework required",
    "QC FAILED - stitching/weave defects found, sent back to finishing",
]

NUM_ORDERS = 40
STUCK_DYEING_IDX = 5      # anomaly 1
QC_FAIL_IDX = 12          # anomaly 3
STUCK_WEAVING_IDX = 20    # anomaly 4


def make_order(order_id, current_stage_idx, current_stage_days, material):
    """Build one order + its stage_history + production_logs rows.

    current_stage_days is how long ago the order entered its current stage --
    built backward from now, so it's the actual knob for "stuck" anomalies
    (unlike computing forward from a random start date, which leaves the
    current stage's entered_at unrelated to the requested duration).
    """
    spans = [None] * (current_stage_idx + 1)
    current_entered_at = datetime.now() - timedelta(days=current_stage_days, hours=random.randint(0, 8))
    spans[current_stage_idx] = (current_entered_at, None)
    exited_at = current_entered_at
    for i in range(current_stage_idx - 1, -1, -1):
        duration_days = max(1, STAGE_AVG_DAYS[STAGES[i]] + random.randint(-1, 1))
        entered_at = exited_at - timedelta(days=duration_days, hours=random.randint(0, 8))
        spans[i] = (entered_at, exited_at)
        exited_at = entered_at

    stage_rows = []
    log_rows = []
    for i in range(current_stage_idx + 1):
        stage = STAGES[i]
        entered_at, stage_exited_at = spans[i]
        stage_rows.append((order_id, stage, entered_at, stage_exited_at))

        # 1-2 production log entries per stage; occasionally a messy/informal one.
        log_time = entered_at + timedelta(hours=random.randint(1, 6))
        note = NEAT_NOTES[stage] if random.random() > 0.25 else random.choice(MESSY_NOTES)
        log_rows.append((order_id, stage, log_time, note))
        if random.random() < 0.2:
            log_rows.append((order_id, stage, log_time + timedelta(hours=3), random.choice(MESSY_NOTES)))

    current_stage = STAGES[current_stage_idx]
    order_start = spans[0][0]
    stage_entered_date = spans[current_stage_idx][0].date()
    total_avg_days = sum(STAGE_AVG_DAYS.values())
    expected_delivery_date = (order_start + timedelta(days=total_avg_days)).date()

    order = {
        "buyer": fake.company(),
        "material": material,
        "quantity": random.randint(200, 3000),
        "current_stage": current_stage,
        "stage_entered_date": stage_entered_date,
        "expected_delivery_date": expected_delivery_date,
    }
    return order, stage_rows, log_rows


def populate(conn) -> dict:
    """Wipe and reseed orders/inventory/production_logs/stage_history on the
    given connection (whatever schema its search_path points at), including
    5 deliberate anomalies.

    Does not commit or close `conn` -- the caller manages that (a plain
    script commits once at the end; a test fixture may run autocommit).

    Returns a dict identifying the 5 seeded anomalies (order_ids and
    material_ids), so callers -- including tests -- don't have to hardcode
    them:
        stuck_dyeing_order_id, stuck_weaving_order_id, qc_fail_order_id,
        mismatch_material_id, mismatch_material_name,
        depleted_material_id, depleted_material_name
    """
    cur = conn.cursor()
    cur.execute("TRUNCATE production_logs, stage_history, orders, inventory RESTART IDENTITY CASCADE;")

    orders_data = []  # (order_id, order dict, stage_rows, log_rows)
    for i in range(NUM_ORDERS):
        material = random.choice(MATERIALS)
        current_stage_idx = random.randint(0, len(STAGES) - 1)
        current_stage_days = max(1, STAGE_AVG_DAYS[STAGES[current_stage_idx]] + random.randint(-1, 2))

        if i == STUCK_DYEING_IDX:
            current_stage_idx = STAGES.index("dyeing")
            current_stage_days = 22  # avg for dyeing is ~3 days -- way out of line
        elif i == STUCK_WEAVING_IDX:
            current_stage_idx = STAGES.index("weaving")
            current_stage_days = 18  # avg for weaving is ~4 days
        elif i == QC_FAIL_IDX:
            current_stage_idx = STAGES.index("QC")
            current_stage_days = 2

        order_id = i + 1
        order, stage_rows, log_rows = make_order(order_id, current_stage_idx, current_stage_days, material)

        if i == STUCK_WEAVING_IDX:
            # Different root cause than the dyeing anomaly: a machine/parts problem, not a QC/color issue.
            log_rows.append(
                (order_id, "weaving", datetime.now() - timedelta(days=1),
                 "loom 3 breakdown, waiting on replacement part from vendor -- no ETA yet")
            )
        if i == QC_FAIL_IDX:
            for note in QC_FAIL_NOTES:
                log_rows.append((order_id, "QC", datetime.now() - timedelta(hours=random.randint(1, 40)), note))

        orders_data.append((order_id, order, stage_rows, log_rows))

    for order_id, order, stage_rows, log_rows in orders_data:
        cur.execute(
            """INSERT INTO orders (order_id, buyer, material, quantity, current_stage,
                                    stage_entered_date, expected_delivery_date)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (order_id, order["buyer"], order["material"], order["quantity"], order["current_stage"],
             order["stage_entered_date"], order["expected_delivery_date"]),
        )
        cur.executemany(
            "INSERT INTO stage_history (order_id, stage, entered_at, exited_at) VALUES (%s, %s, %s, %s)",
            stage_rows,
        )
        cur.executemany(
            "INSERT INTO production_logs (order_id, stage, timestamp, notes) VALUES (%s, %s, %s, %s)",
            log_rows,
        )

    # --- Inventory: a handful of materials, 2-4 batches each (~15-20 rows total). ---
    LOCATIONS = ["Warehouse A", "Warehouse B", "Dye Shed Store", "Finishing Store"]
    inventory_rows = []  # (material_id, material_name, batch_number, quantity_available, location)
    batch_counter = 1
    for material_name in MATERIALS:
        for _ in range(random.randint(2, 4)):
            material_id = f"MAT-{batch_counter:03d}"
            inventory_rows.append([
                material_id, material_name, f"BATCH-{fake.bothify('??###').upper()}",
                round(random.uniform(200, 2000), 1), random.choice(LOCATIONS),
            ])
            batch_counter += 1

    # Demand per material = quantity from orders still in early stages (spinning/weaving/dyeing),
    # i.e. stock that hasn't been consumed yet but will be needed soon.
    upcoming_demand = {}
    for _, order, _, _ in orders_data:
        if order["current_stage"] in ("spinning", "weaving", "dyeing"):
            upcoming_demand[order["material"]] = upcoming_demand.get(order["material"], 0) + order["quantity"]

    # anomaly 5: nearly-depleted batch of the material with the highest upcoming demand.
    top_demand_material = max(upcoming_demand, key=upcoming_demand.get)
    depleted_batch = next(row for row in inventory_rows if row[1] == top_demand_material)
    depleted_batch[3] = 12.0  # units on hand vs. hundreds/thousands of units of upcoming demand

    # anomaly 2: a batch whose recorded quantity is implausibly high given how much of that
    # material orders have already drawn on (consumed = orders past spinning, i.e. weaving onward).
    consumed = {}
    for _, order, _, _ in orders_data:
        if order["current_stage"] != "spinning":
            consumed[order["material"]] = consumed.get(order["material"], 0) + order["quantity"]
    mismatch_material = max(consumed, key=consumed.get)
    mismatch_batch = next(row for row in inventory_rows if row[1] == mismatch_material and row is not depleted_batch)
    mismatch_batch[3] = 9999.0  # suspiciously high/round, doesn't reconcile with consumption above

    cur.executemany(
        """INSERT INTO inventory (material_id, material_name, batch_number, quantity_available, location)
           VALUES (%s, %s, %s, %s, %s)""",
        inventory_rows,
    )
    cur.close()

    return {
        "stuck_dyeing_order_id": STUCK_DYEING_IDX + 1,
        "stuck_weaving_order_id": STUCK_WEAVING_IDX + 1,
        "qc_fail_order_id": QC_FAIL_IDX + 1,
        "mismatch_material_id": mismatch_batch[0],
        "mismatch_material_name": mismatch_material,
        "depleted_material_id": depleted_batch[0],
        "depleted_material_name": top_demand_material,
        "num_inventory_rows": len(inventory_rows),
    }


def main():
    conn = get_connection()
    anomalies = populate(conn)
    conn.commit()
    conn.close()

    print(f"Inserted {NUM_ORDERS} orders and {anomalies['num_inventory_rows']} inventory batches.\n")
    print("Seeded anomalies:")
    print(f"  1. Order stuck in 'dyeing' (order_id={anomalies['stuck_dyeing_order_id']}, ~22 days vs ~3 day avg)")
    print(f"  2. Inventory quantity mismatch vs consumption (material_id={anomalies['mismatch_material_id']}, "
          f"'{anomalies['mismatch_material_name']}')")
    print(f"  3. Order with repeated QC failures (order_id={anomalies['qc_fail_order_id']}, "
          f"{len(QC_FAIL_NOTES)} QC FAILED notes)")
    print(f"  4. Order stuck in 'weaving' from a different cause -- loom breakdown "
          f"(order_id={anomalies['stuck_weaving_order_id']}, ~18 days vs ~4 day avg)")
    print(f"  5. Inventory batch nearly depleted vs upcoming demand (material_id={anomalies['depleted_material_id']}, "
          f"'{anomalies['depleted_material_name']}')")


if __name__ == "__main__":
    main()
