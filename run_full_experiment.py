"""
Run the full 60+ arithmetic experiment.
Simplified version with reduced parameters for feasible runtime.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, nstr
from arithmetics.experiments.population import get_all_arithmetics, summary as pop_summary
from arithmetics.core.defect import DefectMatrix
from arithmetics.core.cohomology import classify_defect
from arithmetics.zeta.dirichlet import zeta_phi

# Reduced parameters for feasible runtime
CONFIG = {
    'precision_defect': 30,
    'precision_zeta': 20,
    'defect_range': (2, 6),  # Smaller range
    'zero_search': {
        're_range': (0.3, 0.7),
        'im_range': (5.0, 20.0),  # Reduced range
        'grid_size': 0.15,  # Coarser grid
    },
    'n_max_zeta': 1000,  # Reduced series terms
    'refinement_precision': 25,
    'max_refine_iter': 20,
}


def grid_search_zeros(phi, config):
    """Quick grid search for zeros."""
    phi.precision = config['precision_zeta']
    re_min, re_max = config['zero_search']['re_range']
    im_min, im_max = config['zero_search']['im_range']
    grid_size = config['zero_search']['grid_size']

    candidates = []
    re = re_min
    while re <= re_max:
        im = im_min
        while im <= im_max:
            s = mpc(re, im)
            try:
                z = zeta_phi(s, phi, n_max=config['n_max_zeta'],
                           precision=config['precision_zeta'], adaptive=True)
                mag = float(fabs(z))
                if mag < 0.5:
                    candidates.append({'re': re, 'im': im, 'mag': mag})
            except:
                pass
            im += grid_size
        re += grid_size

    return sorted(candidates, key=lambda x: x['mag'])[:5]  # Top 5


def refine_zero(phi, re0, im0, config):
    """Simple Newton-Raphson refinement."""
    precision = config['refinement_precision']
    phi.precision = precision
    mp.dps = precision

    s = mpc(re0, im0)
    tol = mp.mpf(10) ** (-(precision - 8))

    for _ in range(config['max_refine_iter']):
        z = zeta_phi(s, phi, n_max=config['n_max_zeta'] * 2, precision=precision, adaptive=True)
        mag = fabs(z)

        if mag < tol:
            return {'s': s, 'mag': float(mag), 'converged': True}

        h = mp.mpf(10) ** (-precision // 3)
        z_re = zeta_phi(s + h, phi, n_max=config['n_max_zeta'] * 2, precision=precision, adaptive=True)
        dz = (z_re - z) / h

        if fabs(dz) < mp.mpf(10) ** (-precision):
            break

        step = z / dz
        if fabs(step) > 0.3:
            step = step * (0.3 / fabs(step))

        s = s - step

        # Keep in bounds
        if s.real < 0.1:
            s = mpc(0.1, s.imag)
        if s.real > 1.5:
            s = mpc(1.5, s.imag)

    z_final = zeta_phi(s, phi, n_max=config['n_max_zeta'] * 2, precision=precision, adaptive=True)
    return {'s': s, 'mag': float(fabs(z_final)), 'converged': False}


def run_single_arithmetic(phi, category, config, index, total):
    """Run experiment on a single arithmetic."""
    start_time = time.time()

    result = {
        'name': phi.name,
        'category': category,
        'index': index,
        'success': False,
        'error': None,
    }

    try:
        # 1. Compute defect
        mp.dps = config['precision_defect']
        phi.precision = config['precision_defect']

        defect_matrix = DefectMatrix(
            phi,
            a_range=config['defect_range'],
            b_range=config['defect_range'],
            precision=config['precision_defect']
        )

        defect_summary = defect_matrix.summary()
        result['defect'] = {
            'max': float(defect_summary['max_defect']),
            'mean': float(defect_summary['mean_defect']),
            'is_zero': defect_summary['is_zero_defect'],
        }

        # 2. Classify defect (cohomology)
        classification = classify_defect(phi, precision=config['precision_defect'])
        result['cohomology'] = {
            'regular_norm': float(classification['regular_norm']) if classification['regular_norm'] else None,
            'cocycle_verified': classification['cocycle_identity_satisfied'],
        }

        # 3. Search for zeros
        candidates = grid_search_zeros(phi, config)
        result['zero_candidates'] = len(candidates)

        zeros_found = []
        if candidates:
            # Refine best candidate
            best = candidates[0]
            refined = refine_zero(phi, best['re'], best['im'], config)

            if refined['converged'] or refined['mag'] < 1e-6:
                zeros_found.append({
                    'real': float(refined['s'].real),
                    'imag': float(refined['s'].imag),
                    'residual': refined['mag'],
                    'converged': refined['converged'],
                })

        result['zeros'] = zeros_found
        result['n_zeros'] = len(zeros_found)
        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    result['time_sec'] = time.time() - start_time

    # Progress update
    status = "OK" if result['success'] else "FAIL"
    zeros_str = f"{result.get('n_zeros', 0)} zeros" if result['success'] else result.get('error', 'unknown')[:30]
    print(f"[{index:2d}/{total}] {phi.name[:35]:35s} | {status:4s} | {zeros_str} | {result['time_sec']:.1f}s")

    return result


def main():
    print("=" * 80)
    print("FULL 60+ ARITHMETIC EXPERIMENT")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get all arithmetics
    all_arithmetics = get_all_arithmetics()
    pop = pop_summary()

    print(f"Total arithmetics: {pop['total']}")
    for cat, count in pop['by_category'].items():
        print(f"  {cat}: {count}")
    print()

    # Create results directory
    results_dir = Path("results/full_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    print("-" * 80)
    print(f"{'#':>4} {'Arithmetic':<35} | {'Status':4} | {'Result':<20} | Time")
    print("-" * 80)

    all_results = []
    total = len(all_arithmetics)

    for i, (phi, category) in enumerate(all_arithmetics, 1):
        result = run_single_arithmetic(phi, category, CONFIG, i, total)
        all_results.append(result)

        # Save intermediate results every 10 arithmetics
        if i % 10 == 0:
            with open(results_dir / "results_intermediate.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # Final save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(results_dir / f"results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(results_dir / "results_latest.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]

    total_zeros = sum(r.get('n_zeros', 0) for r in successful)
    total_time = sum(r.get('time_sec', 0) for r in all_results)

    print(f"Successful: {len(successful)}/{total}")
    print(f"Failed: {len(failed)}")
    print(f"Total zeros found: {total_zeros}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # By category
    print()
    print("By category:")
    by_cat = {}
    for r in all_results:
        cat = r['category']
        if cat not in by_cat:
            by_cat[cat] = {'total': 0, 'success': 0, 'zeros': 0}
        by_cat[cat]['total'] += 1
        if r['success']:
            by_cat[cat]['success'] += 1
            by_cat[cat]['zeros'] += r.get('n_zeros', 0)

    for cat, stats in by_cat.items():
        print(f"  {cat}: {stats['success']}/{stats['total']} ok, {stats['zeros']} zeros")

    # List zeros found
    print()
    print("Zeros found:")
    for r in all_results:
        if r.get('zeros'):
            for z in r['zeros']:
                status = "CONFIRMED" if z['converged'] else "likely"
                print(f"  {r['name']}: s = {z['real']:.6f} + {z['imag']:.6f}i [{status}]")

    if failed:
        print()
        print("Failed arithmetics:")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', 'unknown')[:50]}")

    print()
    print(f"Results saved to: {results_dir}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
