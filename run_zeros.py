"""
Run zero finding across all 60 arithmetics.
"""

import sys
import time
from datetime import datetime
from typing import List, Tuple

# Add the simulation directory to path
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp
from arithmetics.experiments.population import get_all_arithmetics, summary
from arithmetics.zeta.zeros import ZeroFinder, ZeroSearchRegion

def run_zero_finding():
    """Run zero finding on all 60 arithmetics."""

    print("=" * 70)
    print("TWISTED ZETA FUNCTIONS: ZERO FINDING ACROSS 60 ARITHMETICS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get all arithmetics
    all_arithmetics = get_all_arithmetics()
    pop_summary = summary()

    print(f"Total arithmetics: {pop_summary['total']}")
    for cat, count in pop_summary['by_category'].items():
        print(f"  {cat}: {count}")
    print()

    # Configuration - use smaller parameters for faster execution
    precision = 12
    n_max = 5000  # Reduced for speed
    im_max = 20.0  # Search up to Im(s) = 20

    # Define search region near critical line
    region = ZeroSearchRegion(
        re_min=0.4,
        re_max=0.6,
        im_min=5.0,
        im_max=im_max,
        re_step=0.1,
        im_step=2.0
    )

    print(f"Search parameters:")
    print(f"  Precision: {precision} decimal places")
    print(f"  Max terms: {n_max}")
    print(f"  Region: Re in [{region.re_min}, {region.re_max}], Im in [{region.im_min}, {region.im_max}]")
    print()
    print("-" * 70)

    results = []
    total_zeros = 0
    total_time = 0

    for i, (phi, category) in enumerate(all_arithmetics, 1):
        print(f"\n[{i:2d}/60] {phi.name}")
        print(f"        Category: {category}")

        start_time = time.time()

        try:
            mp.dps = precision
            phi.precision = precision

            finder = ZeroFinder(phi, precision=precision, n_max=n_max)
            zeros = finder.find_in_region(region, tol=1e-10)

            elapsed = time.time() - start_time
            total_time += elapsed

            # Get statistics
            stats = finder.statistics()
            n_zeros = stats.get('n_zeros', 0)
            n_on_critical = stats.get('n_on_critical_line', 0)

            total_zeros += n_zeros

            # Store result
            result = {
                'name': phi.name,
                'category': category,
                'n_zeros': n_zeros,
                'n_on_critical': n_on_critical,
                'zeros': [(z.zero.real, z.zero.imag) for z in zeros],
                'time': elapsed,
                'success': True
            }
            results.append(result)

            print(f"        Found: {n_zeros} zeros ({n_on_critical} on critical line)")
            print(f"        Time: {elapsed:.2f}s")

            # Show first few zeros
            if zeros:
                print(f"        First zeros:")
                for z in zeros[:3]:
                    print(f"          s = {z.zero.real:.6f} + {z.zero.imag:.6f}i")
                if len(zeros) > 3:
                    print(f"          ... and {len(zeros) - 3} more")

        except Exception as e:
            elapsed = time.time() - start_time
            total_time += elapsed

            result = {
                'name': phi.name,
                'category': category,
                'n_zeros': 0,
                'n_on_critical': 0,
                'zeros': [],
                'time': elapsed,
                'success': False,
                'error': str(e)
            }
            results.append(result)

            print(f"        ERROR: {e}")
            print(f"        Time: {elapsed:.2f}s")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Total zeros found: {total_zeros}")
    print()

    # Group by category
    by_category = {}
    for r in results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = {'n_arithmetics': 0, 'n_zeros': 0, 'n_on_critical': 0, 'n_success': 0}
        by_category[cat]['n_arithmetics'] += 1
        by_category[cat]['n_zeros'] += r['n_zeros']
        by_category[cat]['n_on_critical'] += r['n_on_critical']
        if r['success']:
            by_category[cat]['n_success'] += 1

    print("Results by category:")
    print("-" * 70)
    print(f"{'Category':<20} {'Arithmetics':>12} {'Zeros':>10} {'On Critical':>12} {'Success':>10}")
    print("-" * 70)

    for cat, stats in by_category.items():
        print(f"{cat:<20} {stats['n_arithmetics']:>12} {stats['n_zeros']:>10} {stats['n_on_critical']:>12} {stats['n_success']:>10}")

    print("-" * 70)
    total_success = sum(1 for r in results if r['success'])
    total_on_critical = sum(r['n_on_critical'] for r in results)
    print(f"{'TOTAL':<20} {len(results):>12} {total_zeros:>10} {total_on_critical:>12} {total_success:>10}")

    # Show arithmetics with most zeros
    print()
    print("Top 10 arithmetics by zeros found:")
    print("-" * 70)
    sorted_results = sorted(results, key=lambda x: x['n_zeros'], reverse=True)
    for r in sorted_results[:10]:
        print(f"  {r['name']:<45} {r['n_zeros']:>3} zeros")

    # Show zeros off critical line (if any)
    off_critical = [(r['name'], z) for r in results for z in r['zeros'] if abs(z[0] - 0.5) > 0.01]
    if off_critical:
        print()
        print(f"Zeros OFF critical line (|Re(s) - 0.5| > 0.01): {len(off_critical)}")
        print("-" * 70)
        for name, (re, im) in off_critical[:20]:
            print(f"  {name:<40} s = {re:.6f} + {im:.6f}i")
        if len(off_critical) > 20:
            print(f"  ... and {len(off_critical) - 20} more")
    else:
        print()
        print("All zeros found are on or near the critical line Re(s) = 0.5")

    return results


if __name__ == "__main__":
    results = run_zero_finding()
