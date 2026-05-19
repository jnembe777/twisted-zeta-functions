"""
Fast zero finding across all arithmetics - uses minimal computation.
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs
from arithmetics.experiments.population import get_all_arithmetics, summary
from arithmetics.zeta.dirichlet import zeta_phi

def quick_zero_scan(phi, precision=10, n_max=2000):
    """
    Quick scan for zeros near the critical line.
    Evaluates at known Riemann zero locations and nearby.
    """
    mp.dps = precision
    phi.precision = precision

    # Check at some known Riemann zero imaginary parts
    test_im = [14.13, 21.02, 25.01, 30.42, 32.94]
    test_re = [0.5, 0.45, 0.55]

    candidates = []

    for im in test_im:
        for re in test_re:
            s = mpc(re, im)
            try:
                z = zeta_phi(s, phi, n_max=n_max, precision=precision, adaptive=True)
                mag = float(fabs(z))
                if mag < 1.0:  # Potential zero nearby
                    candidates.append((re, im, mag))
            except:
                pass

    return candidates

def run_fast_zero_finding():
    """Run fast zero finding on all arithmetics."""

    print("=" * 70)
    print("TWISTED ZETA FUNCTIONS: FAST ZERO SURVEY")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_arithmetics = get_all_arithmetics()
    pop_summary = summary()

    print(f"Total arithmetics: {pop_summary['total']}")
    print(f"Testing at Im(s) in [14.13, 21.02, 25.01, 30.42, 32.94]")
    print(f"Precision: 10 decimal places, Max terms: 2000")
    print()
    print("-" * 70)

    results = []
    total_candidates = 0
    start_total = time.time()

    for i, (phi, category) in enumerate(all_arithmetics, 1):
        start_time = time.time()

        try:
            candidates = quick_zero_scan(phi, precision=10, n_max=2000)
            elapsed = time.time() - start_time

            n_candidates = len(candidates)
            total_candidates += n_candidates

            result = {
                'name': phi.name,
                'category': category,
                'candidates': candidates,
                'n_candidates': n_candidates,
                'time': elapsed,
                'success': True
            }
            results.append(result)

            status = f"{n_candidates} candidates" if n_candidates > 0 else "no candidates"
            print(f"[{i:2d}/{pop_summary['total']}] {phi.name[:40]:<40} {status:<15} ({elapsed:.1f}s)")

            # Show candidates if found
            if candidates:
                for re, im, mag in candidates[:2]:
                    print(f"         -> s = {re:.2f} + {im:.2f}i, |zeta| = {mag:.2e}")

        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                'name': phi.name,
                'category': category,
                'candidates': [],
                'n_candidates': 0,
                'time': elapsed,
                'success': False,
                'error': str(e)
            }
            results.append(result)
            print(f"[{i:2d}/{pop_summary['total']}] {phi.name[:40]:<40} ERROR ({elapsed:.1f}s)")

    total_time = time.time() - start_total

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Total zero candidates found: {total_candidates}")
    print()

    # Group by category
    by_category = {}
    for r in results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = {'n_arithmetics': 0, 'n_candidates': 0, 'n_success': 0}
        by_category[cat]['n_arithmetics'] += 1
        by_category[cat]['n_candidates'] += r['n_candidates']
        if r['success']:
            by_category[cat]['n_success'] += 1

    print("Results by category:")
    print("-" * 70)
    print(f"{'Category':<20} {'Arithmetics':>12} {'Candidates':>12} {'Success':>10}")
    print("-" * 70)

    for cat, stats in by_category.items():
        print(f"{cat:<20} {stats['n_arithmetics']:>12} {stats['n_candidates']:>12} {stats['n_success']:>10}")

    print("-" * 70)
    total_success = sum(1 for r in results if r['success'])
    print(f"{'TOTAL':<20} {len(results):>12} {total_candidates:>12} {total_success:>10}")

    # Show top arithmetics by candidates
    print()
    print("Arithmetics with most zero candidates:")
    print("-" * 70)
    sorted_results = sorted(results, key=lambda x: x['n_candidates'], reverse=True)
    for r in sorted_results[:15]:
        if r['n_candidates'] > 0:
            print(f"  {r['name']:<50} {r['n_candidates']:>3} candidates")

    return results


if __name__ == "__main__":
    results = run_fast_zero_finding()
