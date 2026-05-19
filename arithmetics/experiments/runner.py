"""
Parallel execution orchestrator for twisted zeta experiments.

Manages the execution of experiments across multiple arithmetics,
with progress tracking, checkpointing, and database storage.
"""

import os
import json
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import yaml
from mpmath import mp

from arithmetics.core.transfer import ArithmeticTransfer
from arithmetics.core.defect import DefectMatrix
from arithmetics.core.cohomology import classify_defect, CohomologyAnalyzer
from arithmetics.zeta.dirichlet import zeta_phi, DirichletSeriesEvaluator
from arithmetics.zeta.zeros import ZeroFinder, ZeroSearchRegion, ZeroResult
from arithmetics.analytics.chebyshev import ChebyshevAnalyzer
from arithmetics.analytics.distribution import DistributionAnalyzer


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    precision_defect: int = 50
    precision_zeta: int = 30
    precision_zeros: int = 20
    n_max: int = 1000000
    newton_max_iter: int = 100
    newton_tolerance: float = 1e-20
    defect_a_range: Tuple[int, int] = (2, 10)
    defect_b_range: Tuple[int, int] = (2, 10)
    zero_search_im_max: float = 50.0
    chebyshev_x_max: int = 100000
    distribution_X: int = 10000


@dataclass
class ExperimentResult:
    """Result of a single arithmetic experiment."""
    arithmetic_name: str
    category: str
    success: bool
    error_message: Optional[str]

    # Defect results
    max_defect: Optional[float]
    mean_defect: Optional[float]
    regular_norm: Optional[float]
    is_zero_defect: Optional[bool]
    cocycle_verified: Optional[bool]

    # Zero results
    n_zeros_found: int
    zeros_on_critical: int
    zeros_off_critical: int
    mean_zero_re: Optional[float]
    std_zero_re: Optional[float]

    # Distribution results
    delta_distribution: Optional[float]

    # Timing
    computation_time_sec: float


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(
    phi: ArithmeticTransfer,
    category: str,
    config: ExperimentConfig
) -> ExperimentResult:
    """
    Run a complete experiment for a single arithmetic transfer.

    Args:
        phi: The arithmetic transfer to study
        category: Category name (Hierarchy, CoherentTwist, Exponential, Arbitrary)
        config: Experiment configuration

    Returns:
        ExperimentResult with all computed metrics
    """
    start_time = time.time()

    try:
        # 1. Compute defect and cohomology
        phi.precision = config.precision_defect
        mp.dps = config.precision_defect

        defect_matrix = DefectMatrix(
            phi,
            a_range=config.defect_a_range,
            b_range=config.defect_b_range,
            precision=config.precision_defect
        )

        classification = classify_defect(phi, precision=config.precision_defect)

        # 2. Find zeros
        phi.precision = config.precision_zeros
        mp.dps = config.precision_zeros

        zero_finder = ZeroFinder(phi, precision=config.precision_zeros, n_max=config.n_max)

        # Search near critical line
        critical_region = ZeroSearchRegion(
            re_min=0.4, re_max=0.6,
            im_min=0.0, im_max=config.zero_search_im_max,
            re_step=0.02, im_step=0.5
        )
        zeros = zero_finder.find_in_region(critical_region, tol=config.newton_tolerance)

        zero_stats = zero_finder.statistics()

        # 3. Compute distribution defect (only for exponential transfers with defect)
        delta_dist = None
        if classification['max_defect'] > 1e-30:
            try:
                dist_analyzer = DistributionAnalyzer(phi, precision=20)
                delta_result = dist_analyzer.full_analysis(min(config.distribution_X, 1000))
                delta_dist = delta_result['delta_analysis']['delta_phi_X']
            except Exception:
                delta_dist = None

        computation_time = time.time() - start_time

        return ExperimentResult(
            arithmetic_name=phi.name,
            category=category,
            success=True,
            error_message=None,
            max_defect=classification['max_defect'],
            mean_defect=classification['mean_defect'],
            regular_norm=classification['regular_norm'],
            is_zero_defect=classification['is_zero_defect'],
            cocycle_verified=classification['cocycle_identity_satisfied'],
            n_zeros_found=zero_stats.get('n_zeros', 0),
            zeros_on_critical=zero_stats.get('n_on_critical_line', 0),
            zeros_off_critical=zero_stats.get('n_off_critical_line', 0),
            mean_zero_re=zero_stats.get('mean_re'),
            std_zero_re=zero_stats.get('std_re'),
            delta_distribution=delta_dist,
            computation_time_sec=computation_time
        )

    except Exception as e:
        computation_time = time.time() - start_time
        return ExperimentResult(
            arithmetic_name=phi.name,
            category=category,
            success=False,
            error_message=str(e),
            max_defect=None,
            mean_defect=None,
            regular_norm=None,
            is_zero_defect=None,
            cocycle_verified=None,
            n_zeros_found=0,
            zeros_on_critical=0,
            zeros_off_critical=0,
            mean_zero_re=None,
            std_zero_re=None,
            delta_distribution=None,
            computation_time_sec=computation_time
        )


def _run_experiment_wrapper(args: Tuple[ArithmeticTransfer, str, ExperimentConfig]) -> ExperimentResult:
    """Wrapper for multiprocessing."""
    phi, category, config = args
    return run_experiment(phi, category, config)


class ExperimentRunner:
    """
    Orchestrates parallel execution of experiments across multiple arithmetics.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        db_path: Optional[str] = None,
        n_workers: Optional[int] = None
    ):
        """
        Initialize the experiment runner.

        Args:
            config_path: Path to configuration YAML
            db_path: Path to SQLite database
            n_workers: Number of parallel workers
        """
        self.config = load_config(config_path)
        self.db_path = db_path or self.config.get('database', {}).get('path', './data/arithmetics.db')
        self.n_workers = n_workers or self.config.get('parallel', {}).get('n_workers', 4)

        self.results: List[ExperimentResult] = []
        self.run_id: Optional[int] = None

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_experiment_config(self) -> ExperimentConfig:
        """Build experiment config from loaded YAML."""
        precision = self.config.get('precision', {})
        search = self.config.get('search', {})
        defect = self.config.get('defect', {})
        chebyshev = self.config.get('chebyshev', {})
        distribution = self.config.get('distribution', {})

        return ExperimentConfig(
            precision_defect=precision.get('defect', 50),
            precision_zeta=precision.get('zeta', 30),
            precision_zeros=precision.get('zeros', 20),
            n_max=search.get('n_max', 1000000),
            newton_max_iter=search.get('newton_max_iter', 100),
            newton_tolerance=search.get('newton_tolerance', 1e-20),
            defect_a_range=(defect.get('a_min', 2), defect.get('a_max', 10)),
            defect_b_range=(defect.get('b_min', 2), defect.get('b_max', 10)),
            zero_search_im_max=search.get('regions', {}).get('critical_line', {}).get('im_max', 50.0),
            chebyshev_x_max=chebyshev.get('x_max', 100000),
            distribution_X=distribution.get('X_default', 10000)
        )

    def _init_database(self):
        """Initialize database connection and ensure schema exists."""
        from arithmetics.data import initialize_database

        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create database if it doesn't exist
        if not Path(self.db_path).exists():
            # Read and execute schema
            schema_path = Path(__file__).parent.parent / "data" / "schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema = f.read()

                conn = sqlite3.connect(self.db_path)
                conn.executescript(schema)
                conn.commit()
                conn.close()

    def _start_run(self, n_arithmetics: int) -> int:
        """Record start of experiment run in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiment_runs (run_name, n_arithmetics_total, config_json)
            VALUES (?, ?, ?)
        """, (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            n_arithmetics,
            json.dumps(self.config)
        ))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return run_id

    def _update_run(self, run_id: int, n_completed: int, n_zeros: int, status: str = 'running'):
        """Update run progress in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE experiment_runs
            SET n_arithmetics_completed = ?, n_zeros_found = ?, status = ?
            WHERE id = ?
        """, (n_completed, n_zeros, status, run_id))

        conn.commit()
        conn.close()

    def _save_result(self, result: ExperimentResult, zeros: List[ZeroResult] = None):
        """Save experiment result to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert or get arithmetic metadata
        cursor.execute("""
            INSERT OR IGNORE INTO arithmetic_metadata (name, category, generator_expr, defect_expected, domain)
            VALUES (?, ?, ?, ?, ?)
        """, (
            result.arithmetic_name,
            result.category,
            result.arithmetic_name,
            not result.is_zero_defect if result.is_zero_defect is not None else True,
            'Z*'
        ))

        cursor.execute("SELECT id FROM arithmetic_metadata WHERE name = ?", (result.arithmetic_name,))
        arithmetic_id = cursor.fetchone()[0]

        # Insert defect metrics
        cursor.execute("""
            INSERT OR REPLACE INTO defect_metrics
            (arithmetic_id, max_defect, mean_defect, regular_norm, is_zero_defect, cocycle_verified,
             delta_distribution, computation_time_sec, precision_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            arithmetic_id,
            result.max_defect,
            result.mean_defect,
            result.regular_norm,
            result.is_zero_defect,
            result.cocycle_verified,
            result.delta_distribution,
            result.computation_time_sec,
            50  # precision
        ))

        # Insert zeros if provided
        if zeros:
            for zero in zeros:
                cursor.execute("""
                    INSERT OR IGNORE INTO zero_detections
                    (arithmetic_id, zero_re, zero_im, multiplicity, detection_method, error_estimate)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    arithmetic_id,
                    zero.zero.real,
                    zero.zero.imag,
                    zero.multiplicity,
                    zero.method,
                    zero.error_estimate
                ))

        conn.commit()
        conn.close()

    def run_all(
        self,
        arithmetics: List[Tuple[ArithmeticTransfer, str]],
        parallel: bool = True
    ) -> List[ExperimentResult]:
        """
        Run experiments on all provided arithmetics.

        Args:
            arithmetics: List of (transfer, category) tuples
            parallel: Whether to run in parallel

        Returns:
            List of ExperimentResult objects
        """
        self._init_database()
        exp_config = self._get_experiment_config()

        n_arithmetics = len(arithmetics)
        self.run_id = self._start_run(n_arithmetics)

        self.logger.info(f"Starting experiment run {self.run_id} with {n_arithmetics} arithmetics")

        self.results = []
        total_zeros = 0

        if parallel and self.n_workers > 1:
            # Parallel execution
            args_list = [(phi, cat, exp_config) for phi, cat in arithmetics]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_run_experiment_wrapper, args): args[0].name
                    for args in args_list
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        self._save_result(result)

                        total_zeros += result.n_zeros_found
                        self._update_run(self.run_id, len(self.results), total_zeros)

                        status = "SUCCESS" if result.success else "FAILED"
                        self.logger.info(
                            f"[{len(self.results)}/{n_arithmetics}] {name}: {status} "
                            f"({result.n_zeros_found} zeros, {result.computation_time_sec:.1f}s)"
                        )

                    except Exception as e:
                        self.logger.error(f"Error processing {name}: {e}")

        else:
            # Sequential execution
            for i, (phi, category) in enumerate(arithmetics):
                self.logger.info(f"[{i+1}/{n_arithmetics}] Processing {phi.name}...")

                result = run_experiment(phi, category, exp_config)
                self.results.append(result)
                self._save_result(result)

                total_zeros += result.n_zeros_found
                self._update_run(self.run_id, len(self.results), total_zeros)

                status = "SUCCESS" if result.success else "FAILED"
                self.logger.info(
                    f"  -> {status}: {result.n_zeros_found} zeros, {result.computation_time_sec:.1f}s"
                )

        # Mark run as completed
        self._update_run(self.run_id, len(self.results), total_zeros, 'completed')

        self.logger.info(f"Experiment run {self.run_id} completed: {len(self.results)} arithmetics processed")

        return self.results

    def run_single(
        self,
        phi: ArithmeticTransfer,
        category: str = "Custom"
    ) -> ExperimentResult:
        """
        Run experiment on a single arithmetic.

        Args:
            phi: The arithmetic transfer
            category: Category name

        Returns:
            ExperimentResult
        """
        self._init_database()
        exp_config = self._get_experiment_config()

        self.logger.info(f"Running single experiment on {phi.name}")

        result = run_experiment(phi, category, exp_config)
        self._save_result(result)

        return result

    def summary(self) -> Dict[str, Any]:
        """Generate summary of experiment results."""
        if not self.results:
            return {'error': 'No results available'}

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        # Group by category
        by_category = {}
        for r in successful:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r)

        # Compute statistics
        total_zeros = sum(r.n_zeros_found for r in successful)
        zeros_on_critical = sum(r.zeros_on_critical for r in successful)

        return {
            'n_total': len(self.results),
            'n_successful': len(successful),
            'n_failed': len(failed),
            'total_zeros_found': total_zeros,
            'zeros_on_critical_line': zeros_on_critical,
            'fraction_on_critical': zeros_on_critical / total_zeros if total_zeros > 0 else 0,
            'by_category': {
                cat: {
                    'n_arithmetics': len(results),
                    'n_zeros': sum(r.n_zeros_found for r in results),
                    'n_zero_defect': sum(1 for r in results if r.is_zero_defect),
                    'avg_time': sum(r.computation_time_sec for r in results) / len(results),
                }
                for cat, results in by_category.items()
            },
            'failed_arithmetics': [r.arithmetic_name for r in failed],
            'total_time_sec': sum(r.computation_time_sec for r in self.results),
        }


def run_all(
    arithmetics: List[Tuple[ArithmeticTransfer, str]],
    n_workers: int = 4,
    config_path: Optional[str] = None
) -> List[ExperimentResult]:
    """
    Convenience function to run all experiments.

    Args:
        arithmetics: List of (transfer, category) tuples
        n_workers: Number of parallel workers
        config_path: Path to config YAML

    Returns:
        List of results
    """
    runner = ExperimentRunner(config_path=config_path, n_workers=n_workers)
    return runner.run_all(arithmetics)
