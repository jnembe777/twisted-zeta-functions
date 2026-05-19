"""
Save all experiment results to SQLite database.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Paths
DB_PATH = Path("data/arithmetics.db")
SCHEMA_PATH = Path("arithmetics/data/schema.sql")
RESULTS_PATH = Path("results/full_experiment/results_latest.json")

def create_database():
    """Create database with schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Simplified schema (avoiding expression constraints)
    schema = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS arithmetic_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        generator_expr TEXT NOT NULL,
        defect_expected BOOLEAN NOT NULL,
        domain TEXT DEFAULT 'Z*',
        parameters TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_arithmetic_category ON arithmetic_metadata(category);

    CREATE TABLE IF NOT EXISTS zero_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        arithmetic_id INTEGER NOT NULL REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
        zero_re REAL NOT NULL,
        zero_im REAL NOT NULL,
        multiplicity INTEGER DEFAULT 1,
        detection_method TEXT NOT NULL,
        error_estimate REAL,
        precision_used INTEGER,
        n_iterations INTEGER,
        verified BOOLEAN DEFAULT FALSE,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_zero_arithmetic ON zero_detections(arithmetic_id);
    CREATE INDEX IF NOT EXISTS idx_zero_re ON zero_detections(zero_re);
    CREATE INDEX IF NOT EXISTS idx_zero_im ON zero_detections(zero_im);

    CREATE TABLE IF NOT EXISTS defect_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        arithmetic_id INTEGER UNIQUE NOT NULL REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
        max_defect REAL,
        max_defect_at TEXT,
        mean_defect REAL,
        regular_norm REAL,
        mu_coefficient REAL,
        nu_coefficient REAL,
        projection_residual REAL,
        delta_distribution REAL,
        sigma_0_estimate REAL,
        cocycle_verified BOOLEAN,
        is_zero_defect BOOLEAN,
        h2_dimension INTEGER,
        computation_time_sec REAL,
        precision_used INTEGER,
        computed_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS experiment_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_name TEXT NOT NULL,
        started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        completed_at DATETIME,
        status TEXT DEFAULT 'running',
        n_arithmetics_total INTEGER,
        n_arithmetics_completed INTEGER DEFAULT 0,
        n_zeros_found INTEGER DEFAULT 0,
        config_json TEXT,
        error_message TEXT,
        host_info TEXT
    );

    CREATE VIEW IF NOT EXISTS zeros_near_critical AS
    SELECT
        am.name AS arithmetic_name,
        am.category,
        zd.zero_re,
        zd.zero_im,
        ABS(zd.zero_re - 0.5) AS distance_to_critical,
        zd.error_estimate,
        zd.detection_method
    FROM zero_detections zd
    JOIN arithmetic_metadata am ON zd.arithmetic_id = am.id
    WHERE zd.zero_re BETWEEN 0.0 AND 1.0
    ORDER BY distance_to_critical;

    CREATE VIEW IF NOT EXISTS defect_summary_by_category AS
    SELECT
        am.category,
        COUNT(*) AS n_arithmetics,
        AVG(dm.max_defect) AS avg_max_defect,
        AVG(dm.regular_norm) AS avg_regular_norm,
        SUM(CASE WHEN dm.is_zero_defect THEN 1 ELSE 0 END) AS n_zero_defect,
        SUM(CASE WHEN dm.cocycle_verified THEN 1 ELSE 0 END) AS n_cocycle_verified
    FROM arithmetic_metadata am
    LEFT JOIN defect_metrics dm ON am.id = dm.arithmetic_id
    GROUP BY am.category;

    CREATE VIEW IF NOT EXISTS zero_statistics AS
    SELECT
        am.id AS arithmetic_id,
        am.name AS arithmetic_name,
        am.category,
        COUNT(zd.id) AS n_zeros,
        AVG(zd.zero_re) AS avg_zero_re,
        MIN(zd.zero_re) AS min_zero_re,
        MAX(zd.zero_re) AS max_zero_re,
        AVG(ABS(zd.zero_re - 0.5)) AS avg_distance_to_critical,
        MIN(zd.zero_im) AS min_zero_im,
        MAX(zd.zero_im) AS max_zero_im
    FROM arithmetic_metadata am
    LEFT JOIN zero_detections zd ON am.id = zd.arithmetic_id
    GROUP BY am.id;
    """

    # Create database
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema)
    conn.commit()
    print(f"Database created: {DB_PATH}")
    return conn


def load_results():
    """Load experiment results from JSON."""
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)


def save_results(conn, results):
    """Save all results to database."""
    cursor = conn.cursor()

    # Track statistics
    stats = {
        'arithmetics': 0,
        'zeros': 0,
        'defect_metrics': 0,
    }

    # Create experiment run record
    cursor.execute("""
        INSERT INTO experiment_runs (run_name, n_arithmetics_total, status, config_json)
        VALUES (?, ?, ?, ?)
    """, (
        f"full_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        len(results),
        'completed',
        json.dumps({'source': 'run_full_experiment.py'})
    ))
    run_id = cursor.lastrowid

    for r in results:
        # 1. Insert arithmetic metadata
        defect_expected = not r.get('defect', {}).get('is_zero', True)

        cursor.execute("""
            INSERT OR IGNORE INTO arithmetic_metadata
            (name, category, generator_expr, defect_expected, parameters)
            VALUES (?, ?, ?, ?, ?)
        """, (
            r['name'],
            r['category'],
            r['name'],  # Using name as expression for now
            defect_expected,
            json.dumps({'index': r.get('index')})
        ))

        # Get arithmetic_id
        cursor.execute("SELECT id FROM arithmetic_metadata WHERE name = ?", (r['name'],))
        row = cursor.fetchone()
        if row is None:
            continue
        arithmetic_id = row[0]
        stats['arithmetics'] += 1

        # 2. Insert defect metrics
        if r.get('defect'):
            defect = r['defect']
            cohom = r.get('cohomology', {})

            # Handle infinity and NaN
            max_defect = defect.get('max')
            mean_defect = defect.get('mean')
            regular_norm = cohom.get('regular_norm')

            # Convert infinity to None for database
            if max_defect == float('inf') or (isinstance(max_defect, float) and max_defect != max_defect):
                max_defect = None
            if mean_defect == float('inf') or (isinstance(mean_defect, float) and mean_defect != mean_defect):
                mean_defect = None
            if regular_norm is not None and (regular_norm != regular_norm):  # NaN check
                regular_norm = None

            cursor.execute("""
                INSERT OR REPLACE INTO defect_metrics
                (arithmetic_id, max_defect, mean_defect, regular_norm,
                 is_zero_defect, cocycle_verified, computation_time_sec, precision_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                arithmetic_id,
                max_defect,
                mean_defect,
                regular_norm,
                defect.get('is_zero'),
                cohom.get('cocycle_verified'),
                r.get('time_sec'),
                30  # precision used
            ))
            stats['defect_metrics'] += 1

        # 3. Insert zeros
        if r.get('zeros'):
            for z in r['zeros']:
                cursor.execute("""
                    INSERT OR IGNORE INTO zero_detections
                    (arithmetic_id, zero_re, zero_im, detection_method,
                     error_estimate, precision_used, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    arithmetic_id,
                    z['real'],
                    z['imag'],
                    'newton' if z.get('converged') else 'grid_search',
                    z.get('residual'),
                    25,  # precision used
                    z.get('converged', False)
                ))
                stats['zeros'] += 1

    # Update run record
    cursor.execute("""
        UPDATE experiment_runs
        SET completed_at = CURRENT_TIMESTAMP,
            n_arithmetics_completed = ?,
            n_zeros_found = ?
        WHERE id = ?
    """, (stats['arithmetics'], stats['zeros'], run_id))

    conn.commit()
    return stats


def print_summary(conn):
    """Print database summary."""
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)

    # Count records in each table
    tables = [
        ('arithmetic_metadata', 'Arithmetics'),
        ('zero_detections', 'Zeros'),
        ('defect_metrics', 'Defect Metrics'),
        ('experiment_runs', 'Experiment Runs'),
    ]

    for table, label in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {label}: {count}")

    # Category breakdown
    print("\nBy Category:")
    cursor.execute("""
        SELECT category, COUNT(*) as n
        FROM arithmetic_metadata
        GROUP BY category
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    # Zero statistics
    print("\nZero Statistics:")
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            AVG(zero_re) as avg_re,
            MIN(zero_re) as min_re,
            MAX(zero_re) as max_re,
            SUM(CASE WHEN zero_re BETWEEN 0.45 AND 0.55 THEN 1 ELSE 0 END) as near_critical
        FROM zero_detections
    """)
    row = cursor.fetchone()
    print(f"  Total zeros: {row[0]}")
    print(f"  Mean Re(s): {row[1]:.4f}")
    print(f"  Range Re(s): [{row[2]:.4f}, {row[3]:.4f}]")
    print(f"  Near critical line (±0.05): {row[4]}")

    # Zeros by category
    print("\nZeros by Category:")
    cursor.execute("""
        SELECT am.category, COUNT(zd.id) as n_zeros, AVG(zd.zero_re) as avg_re
        FROM arithmetic_metadata am
        LEFT JOIN zero_detections zd ON am.id = zd.arithmetic_id
        GROUP BY am.category
    """)
    for row in cursor.fetchall():
        avg_re = f"{row[2]:.4f}" if row[2] else "N/A"
        print(f"  {row[0]}: {row[1]} zeros, avg Re(s) = {avg_re}")

    # Defect summary
    print("\nDefect Summary:")
    cursor.execute("""
        SELECT
            SUM(CASE WHEN is_zero_defect THEN 1 ELSE 0 END) as zero_defect,
            SUM(CASE WHEN NOT is_zero_defect THEN 1 ELSE 0 END) as nonzero_defect,
            SUM(CASE WHEN cocycle_verified THEN 1 ELSE 0 END) as cocycle_ok
        FROM defect_metrics
    """)
    row = cursor.fetchone()
    print(f"  Zero defect: {row[0]}")
    print(f"  Non-zero defect: {row[1]}")
    print(f"  Cocycle verified: {row[2]}")


def main():
    print("="*60)
    print("SAVING RESULTS TO DATABASE")
    print("="*60)

    # Create database
    conn = create_database()

    # Load results
    print(f"\nLoading results from: {RESULTS_PATH}")
    results = load_results()
    print(f"Loaded {len(results)} experiment results")

    # Save to database
    print("\nSaving to database...")
    stats = save_results(conn, results)

    print(f"\nInserted:")
    print(f"  Arithmetics: {stats['arithmetics']}")
    print(f"  Zeros: {stats['zeros']}")
    print(f"  Defect metrics: {stats['defect_metrics']}")

    # Print summary
    print_summary(conn)

    conn.close()
    print(f"\nDatabase saved to: {DB_PATH.absolute()}")


if __name__ == "__main__":
    main()
