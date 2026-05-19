-- ===========================================================================
-- Database Schema for Twisted Zeta Functions Experimental Framework
-- ===========================================================================

-- Enable foreign key support
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Table: arithmetic_metadata
-- Stores metadata about each arithmetic transfer being studied
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS arithmetic_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('Hierarchy', 'CoherentTwist', 'Exponential', 'Arbitrary')),
    generator_expr TEXT NOT NULL,       -- Symbolic expression for φ
    defect_expected BOOLEAN NOT NULL,   -- Whether defect is expected to be non-zero
    domain TEXT DEFAULT 'Z*',           -- Domain of definition
    parameters TEXT,                    -- JSON string of transfer parameters
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Index for fast category lookups
CREATE INDEX IF NOT EXISTS idx_arithmetic_category ON arithmetic_metadata(category);

-- ---------------------------------------------------------------------------
-- Table: zero_detections
-- Stores detected zeros of twisted zeta functions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zero_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arithmetic_id INTEGER NOT NULL REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
    zero_re REAL NOT NULL,              -- Real part of the zero
    zero_im REAL NOT NULL,              -- Imaginary part of the zero
    multiplicity INTEGER DEFAULT 1,     -- Multiplicity of the zero
    detection_method TEXT NOT NULL CHECK (detection_method IN ('newton', 'argument_principle', 'grid_search', 'combined')),
    error_estimate REAL,                -- Estimated error in zero location
    precision_used INTEGER,             -- Decimal places used in computation
    n_iterations INTEGER,               -- Number of iterations to converge
    verified BOOLEAN DEFAULT FALSE,     -- Whether independently verified
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Ensure uniqueness for each zero (with tolerance)
    UNIQUE (arithmetic_id, ROUND(zero_re, 12), ROUND(zero_im, 12))
);

-- Indexes for zero analysis
CREATE INDEX IF NOT EXISTS idx_zero_arithmetic ON zero_detections(arithmetic_id);
CREATE INDEX IF NOT EXISTS idx_zero_re ON zero_detections(zero_re);
CREATE INDEX IF NOT EXISTS idx_zero_im ON zero_detections(zero_im);
CREATE INDEX IF NOT EXISTS idx_zero_critical ON zero_detections(arithmetic_id, zero_re) WHERE zero_re BETWEEN 0.4 AND 0.6;

-- ---------------------------------------------------------------------------
-- Table: defect_metrics
-- Stores computed defect-related metrics for each arithmetic
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS defect_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arithmetic_id INTEGER UNIQUE NOT NULL REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
    max_defect REAL,                    -- Maximum |δ_φ(a,b)| over test pairs
    max_defect_at TEXT,                 -- JSON: (a, b) where max occurs
    mean_defect REAL,                   -- Mean |δ_φ(a,b)|
    regular_norm REAL,                  -- ||[δ_φ]|| in H²_reg
    mu_coefficient REAL,                -- μ from projection δ ≈ μα^{a+b} + να^{ab}
    nu_coefficient REAL,                -- ν from projection
    projection_residual REAL,           -- Residual after projection
    delta_distribution REAL,            -- Δ_φ(X) for X = 10^6
    sigma_0_estimate REAL,              -- Estimated abscissa of zero-free region
    cocycle_verified BOOLEAN,           -- Whether cocycle identity was verified
    is_zero_defect BOOLEAN,             -- Whether defect is effectively zero
    h2_dimension INTEGER,               -- Dimension of H²_{φ,reg}
    computation_time_sec REAL,          -- Time taken for computation
    precision_used INTEGER,             -- Decimal places used
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Table: chebyshev_results
-- Stores twisted Chebyshev function evaluations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chebyshev_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arithmetic_id INTEGER NOT NULL REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
    x_value REAL NOT NULL,              -- Point of evaluation
    psi_phi_x REAL NOT NULL,            -- ψ_φ(x) = Σ_{n≤x} Λ(n) φ(n)^{-1}
    explicit_formula_value REAL,        -- Value from explicit formula
    remainder_observed REAL,            -- R(x) = ψ_φ(x) - explicit_formula
    remainder_predicted REAL,           -- Predicted bound on R(x)
    n_terms_used INTEGER,               -- Number of terms in sum
    precision_used INTEGER,
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (arithmetic_id, x_value)
);

-- Index for Chebyshev lookups
CREATE INDEX IF NOT EXISTS idx_chebyshev_arithmetic ON chebyshev_results(arithmetic_id);
CREATE INDEX IF NOT EXISTS idx_chebyshev_x ON chebyshev_results(x_value);

-- ---------------------------------------------------------------------------
-- Table: zeta_evaluations
-- Stores evaluations of ζ_φ(s) on a grid
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zeta_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arithmetic_id INTEGER NOT NULL REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
    s_re REAL NOT NULL,                 -- Real part of s
    s_im REAL NOT NULL,                 -- Imaginary part of s
    zeta_re REAL NOT NULL,              -- Real part of ζ_φ(s)
    zeta_im REAL NOT NULL,              -- Imaginary part of ζ_φ(s)
    abs_zeta REAL,                      -- |ζ_φ(s)|
    n_terms_used INTEGER,               -- Terms used in series
    convergence_rate REAL,              -- Rate of convergence
    precision_used INTEGER,
    method TEXT DEFAULT 'dirichlet',    -- 'dirichlet', 'euler_product', 'accelerated'
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (arithmetic_id, ROUND(s_re, 6), ROUND(s_im, 6))
);

-- Indexes for zeta lookups
CREATE INDEX IF NOT EXISTS idx_zeta_arithmetic ON zeta_evaluations(arithmetic_id);
CREATE INDEX IF NOT EXISTS idx_zeta_s ON zeta_evaluations(s_re, s_im);

-- ---------------------------------------------------------------------------
-- Table: twisted_primes
-- Stores φ-twisted primes for each arithmetic
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS twisted_primes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arithmetic_id INTEGER NOT NULL REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
    prime_value INTEGER NOT NULL,       -- The twisted prime p
    is_classical_prime BOOLEAN,         -- Whether also a classical prime
    phi_p REAL,                         -- φ(p)
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (arithmetic_id, prime_value)
);

-- Index for twisted primes
CREATE INDEX IF NOT EXISTS idx_twisted_primes_arithmetic ON twisted_primes(arithmetic_id);

-- ---------------------------------------------------------------------------
-- Table: experiment_runs
-- Tracks experiment execution history
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    n_arithmetics_total INTEGER,
    n_arithmetics_completed INTEGER DEFAULT 0,
    n_zeros_found INTEGER DEFAULT 0,
    config_json TEXT,                   -- JSON of configuration used
    error_message TEXT,
    host_info TEXT                      -- Information about compute environment
);

-- ---------------------------------------------------------------------------
-- Table: validation_results
-- Stores results of validation tests
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_type TEXT NOT NULL CHECK (test_type IN ('reproducibility', 'precision_convergence', 'classical_validation', 'functional_equation')),
    arithmetic_id INTEGER REFERENCES arithmetic_metadata(id) ON DELETE CASCADE,
    passed BOOLEAN NOT NULL,
    expected_value REAL,
    actual_value REAL,
    deviation REAL,
    tolerance REAL,
    details TEXT,                       -- JSON with detailed results
    tested_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Index for validation lookups
CREATE INDEX IF NOT EXISTS idx_validation_arithmetic ON validation_results(arithmetic_id);
CREATE INDEX IF NOT EXISTS idx_validation_type ON validation_results(test_type);

-- ---------------------------------------------------------------------------
-- Views for common queries
-- ---------------------------------------------------------------------------

-- View: Summary of zeros near critical line
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

-- View: Defect summary by category
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

-- View: Zero statistics by arithmetic
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
