-- Textile mill order/inventory reconciliation schema.

-- One row per customer order as it moves through production.
-- current_stage + stage_entered_date let the agent flag orders stuck
-- past a stage's expected duration, or past expected_delivery_date.
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    buyer TEXT,
    material TEXT,
    quantity INTEGER,
    current_stage TEXT,
    stage_entered_date DATE,
    expected_delivery_date DATE
);

-- Current on-hand stock per material batch. The agent cross-checks this
-- against orders.material/quantity to catch mismatches (e.g. an order
-- needs more raw material than is available).
CREATE TABLE IF NOT EXISTS inventory (
    material_id TEXT PRIMARY KEY,
    material_name TEXT,
    batch_number TEXT,
    quantity_available NUMERIC,
    location TEXT
);

-- Append-only event log of production activity per order (e.g. machine
-- runs, QC checks). Free-text notes field for anything the log producer
-- wants to record; not used for stage timing (see stage_history for that).
CREATE TABLE IF NOT EXISTS production_logs (
    log_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    stage TEXT,
    timestamp TIMESTAMP,
    notes TEXT
);

-- One row per stage an order has passed through, with entered_at/exited_at.
-- exited_at IS NULL while the order is still in that stage. This is what
-- lets the agent compute "how long has this order actually spent in each
-- stage" and compare it to expectations -- orders.current_stage only
-- gives the current snapshot, not the history.
CREATE TABLE IF NOT EXISTS stage_history (
    history_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    stage TEXT,
    entered_at TIMESTAMP,
    exited_at TIMESTAMP
);
