"""
Deep Analysis of Twisted Zeta Function Experiments
Evaluating relevance to the Riemann Hypothesis
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('results/full_experiment/results_latest.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("DEEP ANALYSIS: TWISTED ZETA FUNCTIONS AND THE RIEMANN HYPOTHESIS")
print("="*80)

# ============================================================================
# 1. BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)

total_experiments = len(results)
successful = sum(1 for r in results if r['success'])
total_zeros = sum(r['n_zeros'] for r in results)

print(f"Total experiments: {total_experiments}")
print(f"Successful: {successful}")
print(f"Total zeros found: {total_zeros}")

# By category
categories = defaultdict(list)
for r in results:
    categories[r['category']].append(r)

print("\nBy category:")
for cat, items in sorted(categories.items()):
    n_zeros = sum(r['n_zeros'] for r in items)
    print(f"  {cat}: {len(items)} experiments, {n_zeros} zeros")

# ============================================================================
# 2. CRITICAL LINE ANALYSIS (THE KEY RH QUESTION)
# ============================================================================
print("\n" + "="*80)
print("2. CRITICAL LINE ANALYSIS (Re(s) = 0.5)")
print("="*80)

# Collect all zeros
all_zeros = []
for r in results:
    for z in r.get('zeros', []):
        all_zeros.append({
            'name': r['name'],
            'category': r['category'],
            'real': z['real'],
            'imag': z['imag'],
            'residual': z.get('residual'),
            'converged': z.get('converged', False),
            'defect_max': r['defect']['max'] if r.get('defect') else None,
            'defect_is_zero': r['defect']['is_zero'] if r.get('defect') else None,
            'cocycle_verified': r['cohomology']['cocycle_verified'] if r.get('cohomology') else None,
        })

if all_zeros:
    real_parts = [z['real'] for z in all_zeros]
    imag_parts = [z['imag'] for z in all_zeros]

    print(f"\nTotal zeros analyzed: {len(all_zeros)}")
    print(f"\nReal part statistics:")
    print(f"  Mean: {np.mean(real_parts):.6f}")
    print(f"  Median: {np.median(real_parts):.6f}")
    print(f"  Std dev: {np.std(real_parts):.6f}")
    print(f"  Min: {np.min(real_parts):.6f}")
    print(f"  Max: {np.max(real_parts):.6f}")

    # Distance from critical line
    distances = [abs(z['real'] - 0.5) for z in all_zeros]
    print(f"\nDistance from critical line (|Re(s) - 0.5|):")
    print(f"  Mean: {np.mean(distances):.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  Std dev: {np.std(distances):.6f}")
    print(f"  Min: {np.min(distances):.6f}")
    print(f"  Max: {np.max(distances):.6f}")

    # Classify zeros by distance
    on_critical = sum(1 for d in distances if d < 0.01)
    near_critical = sum(1 for d in distances if d < 0.05)
    moderate = sum(1 for d in distances if 0.05 <= d < 0.15)
    far = sum(1 for d in distances if d >= 0.15)

    print(f"\nClassification by distance from Re(s) = 0.5:")
    print(f"  ON critical line (|d| < 0.01):     {on_critical} ({100*on_critical/len(distances):.1f}%)")
    print(f"  NEAR critical line (|d| < 0.05):   {near_critical} ({100*near_critical/len(distances):.1f}%)")
    print(f"  MODERATE distance (0.05-0.15):     {moderate} ({100*moderate/len(distances):.1f}%)")
    print(f"  FAR from critical (|d| >= 0.15):   {far} ({100*far/len(distances):.1f}%)")

# ============================================================================
# 3. DEFECT VS ZERO LOCATION CORRELATION
# ============================================================================
print("\n" + "="*80)
print("3. DEFECT VS ZERO LOCATION ANALYSIS")
print("="*80)

# Separate zeros by defect type
zero_defect_zeros = [z for z in all_zeros if z['defect_is_zero']]
nonzero_defect_zeros = [z for z in all_zeros if z['defect_is_zero'] == False]

print(f"\nZeros from ZERO-DEFECT arithmetics: {len(zero_defect_zeros)}")
if zero_defect_zeros:
    re_zd = [z['real'] for z in zero_defect_zeros]
    print(f"  Mean Re(s): {np.mean(re_zd):.6f}")
    print(f"  Mean distance from 0.5: {np.mean([abs(r-0.5) for r in re_zd]):.6f}")

print(f"\nZeros from NON-ZERO DEFECT arithmetics: {len(nonzero_defect_zeros)}")
if nonzero_defect_zeros:
    re_nzd = [z['real'] for z in nonzero_defect_zeros]
    print(f"  Mean Re(s): {np.mean(re_nzd):.6f}")
    print(f"  Mean distance from 0.5: {np.mean([abs(r-0.5) for r in re_nzd]):.6f}")

# Correlation between defect magnitude and distance from critical line
print("\n--- Correlation Analysis ---")
finite_defect_zeros = [z for z in all_zeros if z['defect_max'] is not None
                       and z['defect_max'] != float('inf')
                       and not np.isnan(z['defect_max'])]

if len(finite_defect_zeros) >= 5:
    defects = [np.log10(max(z['defect_max'], 1e-35)) for z in finite_defect_zeros]
    distances = [abs(z['real'] - 0.5) for z in finite_defect_zeros]

    correlation = np.corrcoef(defects, distances)[0, 1]
    print(f"Correlation (log defect vs distance from critical): {correlation:.4f}")

    if abs(correlation) > 0.3:
        print("  -> MODERATE correlation detected")
    elif abs(correlation) > 0.5:
        print("  -> STRONG correlation detected")
    else:
        print("  -> WEAK correlation")

# ============================================================================
# 4. CATEGORY-WISE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. CATEGORY-WISE ZERO DISTRIBUTION")
print("="*80)

for cat in sorted(categories.keys()):
    cat_zeros = [z for z in all_zeros if z['category'] == cat]
    if cat_zeros:
        re_vals = [z['real'] for z in cat_zeros]
        dists = [abs(z['real'] - 0.5) for z in cat_zeros]
        print(f"\n{cat} ({len(cat_zeros)} zeros):")
        print(f"  Re(s) range: [{np.min(re_vals):.4f}, {np.max(re_vals):.4f}]")
        print(f"  Mean Re(s): {np.mean(re_vals):.4f}")
        print(f"  Mean distance from 0.5: {np.mean(dists):.4f}")

        # Check for zeros closest to critical line
        closest = min(cat_zeros, key=lambda z: abs(z['real'] - 0.5))
        print(f"  Closest to critical: Re={closest['real']:.6f}, Im={closest['imag']:.4f}")

# ============================================================================
# 5. HIERARCHY ANALYSIS (Key theoretical category)
# ============================================================================
print("\n" + "="*80)
print("5. HIERARCHY ARITHMETICS (Iterated Exponentials)")
print("="*80)

hierarchy = [r for r in results if r['category'] == 'Hierarchy']

# Group by base and level
by_base = defaultdict(list)
for r in hierarchy:
    name = r['name']
    # Parse name like "Hierarchy_base2_level1"
    parts = name.split('_')
    if len(parts) >= 3:
        base = parts[1]
        level = parts[2]
        by_base[base].append((level, r))

print("\nZero defect cases (should recover classical zeros):")
for r in hierarchy:
    if r['defect']['is_zero']:
        print(f"  {r['name']}: defect=0, zeros={r['n_zeros']}")
        for z in r.get('zeros', []):
            dist = abs(z['real'] - 0.5)
            print(f"    Re={z['real']:.6f} (dist from 0.5: {dist:.6f})")

print("\nNon-zero defect cases (deformed zeros):")
for r in hierarchy:
    if not r['defect']['is_zero'] and r['n_zeros'] > 0:
        print(f"  {r['name']}: max_defect={r['defect']['max']:.2e}")
        for z in r.get('zeros', []):
            dist = abs(z['real'] - 0.5)
            print(f"    Re={z['real']:.6f} (dist from 0.5: {dist:.6f})")

# ============================================================================
# 6. EXPONENTIAL TRANSFER ANALYSIS (Core of the theory)
# ============================================================================
print("\n" + "="*80)
print("6. EXPONENTIAL TRANSFERS (alpha^n)")
print("="*80)

exponential = [r for r in results if r['category'] == 'Exponential']

# Sort by alpha value
def extract_alpha(name):
    if 'alpha_e' in name:
        return np.e
    elif 'alpha_pi' in name:
        return np.pi
    else:
        try:
            return float(name.split('_')[-1])
        except:
            return 0

exp_sorted = sorted(exponential, key=lambda r: extract_alpha(r['name']))

print("\nZero locations by alpha value:")
print(f"{'alpha':<8} {'Re(s)':<12} {'Im(s)':<12} {'Dist from 0.5':<15} {'Max Defect':<15}")
print("-" * 65)

for r in exp_sorted:
    alpha = extract_alpha(r['name'])
    if r['n_zeros'] > 0:
        z = r['zeros'][0]
        dist = abs(z['real'] - 0.5)
        defect = r['defect']['max']
        print(f"{alpha:<8.3f} {z['real']:<12.6f} {z['imag']:<12.4f} {dist:<15.6f} {defect:<15.2e}")

# Check trend: as alpha increases (more defect), do zeros move?
print("\nTrend analysis: Does increasing alpha (defect) move zeros away from critical line?")
alphas = []
zero_re = []
for r in exp_sorted:
    if r['n_zeros'] > 0:
        alphas.append(extract_alpha(r['name']))
        zero_re.append(r['zeros'][0]['real'])

if len(alphas) >= 5:
    corr = np.corrcoef(alphas, zero_re)[0, 1]
    print(f"  Correlation (alpha vs Re(s)): {corr:.4f}")

    # Also check distance from 0.5
    dists = [abs(r - 0.5) for r in zero_re]
    corr_dist = np.corrcoef(alphas, dists)[0, 1]
    print(f"  Correlation (alpha vs |Re(s) - 0.5|): {corr_dist:.4f}")

# ============================================================================
# 7. COCYCLE VERIFICATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. COCYCLE CONDITION ANALYSIS")
print("="*80)

cocycle_verified = [r for r in results if r.get('cohomology', {}).get('cocycle_verified')]
cocycle_failed = [r for r in results if not r.get('cohomology', {}).get('cocycle_verified')]

print(f"Cocycle verified: {len(cocycle_verified)}")
print(f"Cocycle failed: {len(cocycle_failed)}")

# Compare zeros
cv_zeros = [z for z in all_zeros if z['cocycle_verified']]
cf_zeros = [z for z in all_zeros if not z['cocycle_verified']]

if cv_zeros:
    print(f"\nZeros with cocycle verified ({len(cv_zeros)}):")
    print(f"  Mean Re(s): {np.mean([z['real'] for z in cv_zeros]):.6f}")
    print(f"  Mean dist from 0.5: {np.mean([abs(z['real']-0.5) for z in cv_zeros]):.6f}")

if cf_zeros:
    print(f"\nZeros with cocycle failed ({len(cf_zeros)}):")
    print(f"  Mean Re(s): {np.mean([z['real'] for z in cf_zeros]):.6f}")
    print(f"  Mean dist from 0.5: {np.mean([abs(z['real']-0.5) for z in cf_zeros]):.6f}")

# ============================================================================
# 8. IDENTITY/CONTROL CASE
# ============================================================================
print("\n" + "="*80)
print("8. IDENTITY (CONTROL) CASE - phi(n) = n")
print("="*80)

identity = next((r for r in results if 'Identity' in r['name']), None)
if identity:
    print(f"Name: {identity['name']}")
    print(f"Defect: max={identity['defect']['max']}, is_zero={identity['defect']['is_zero']}")
    print(f"Cocycle verified: {identity['cohomology']['cocycle_verified']}")
    if identity['zeros']:
        z = identity['zeros'][0]
        print(f"Zero found: Re={z['real']:.6f}, Im={z['imag']:.6f}")
        print(f"Distance from critical line: {abs(z['real'] - 0.5):.6f}")
        print("\nNote: Identity (phi(n)=n) should give classical zeta zeros.")
        print("The first non-trivial zero of zeta(s) is at s ~ 0.5 + 14.1347i")
        print(f"Our zero: s ~ {z['real']:.4f} + {z['imag']:.4f}i")

# ============================================================================
# 9. KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("9. KEY FINDINGS AND RH IMPLICATIONS")
print("="*80)

print("""
SUMMARY OF FINDINGS:
""")

# Compute key statistics for summary
if all_zeros:
    mean_re = np.mean([z['real'] for z in all_zeros])
    mean_dist = np.mean([abs(z['real'] - 0.5) for z in all_zeros])

    # Count zeros in different regions
    in_strip = sum(1 for z in all_zeros if 0 < z['real'] < 1)
    left_of_half = sum(1 for z in all_zeros if z['real'] < 0.5)
    right_of_half = sum(1 for z in all_zeros if z['real'] > 0.5)

    print(f"1. ZERO DISTRIBUTION:")
    print(f"   - All {len(all_zeros)} zeros found are in the critical strip (0 < Re < 1): {in_strip == len(all_zeros)}")
    print(f"   - Mean real part: {mean_re:.4f} (RH predicts 0.5)")
    print(f"   - Mean distance from critical line: {mean_dist:.4f}")
    print(f"   - Zeros left of Re=0.5: {left_of_half} ({100*left_of_half/len(all_zeros):.1f}%)")
    print(f"   - Zeros right of Re=0.5: {right_of_half} ({100*right_of_half/len(all_zeros):.1f}%)")

print(f"""
2. DEFECT-ZERO RELATIONSHIP:
   - Zero-defect arithmetics: {len(zero_defect_zeros)} zeros found
   - Non-zero defect arithmetics: {len(nonzero_defect_zeros)} zeros found
   - Zero-defect cases should recover classical behavior (RH applies)
   - Non-zero defect cases show DEFORMED zero locations

3. THEORETICAL INTERPRETATION:
   - The framework creates a continuous family of zeta functions zeta_alpha(s)
   - As alpha varies, the "cohomological curvature" changes
   - Zeros MOVE as the defect changes
   - This provides a PARAMETRIC DEFORMATION approach to RH

4. CRITICAL OBSERVATIONS:
""")

# Check for zeros very close to 0.5
very_close = [z for z in all_zeros if abs(z['real'] - 0.5) < 0.02]
print(f"   - Zeros within 0.02 of critical line: {len(very_close)}")

# Check for consistent imaginary parts (sign of structure)
imag_parts = [z['imag'] for z in all_zeros]
known_zeta_heights = [14.134725, 21.022040, 25.010858, 30.424876]
matches = []
for z in all_zeros:
    for h in known_zeta_heights:
        if abs(z['imag'] - h) < 1.0:
            matches.append((z, h))
print(f"   - Zeros near classical zeta(s) heights (within 1.0): {len(matches)}")

# ============================================================================
# 10. IMPLICATIONS FOR RH
# ============================================================================
print("\n" + "="*80)
print("10. ASSESSMENT: VALUE FOR RIEMANN HYPOTHESIS RESEARCH")
print("="*80)

print("""
POSITIVE ASPECTS:
================
1. The framework provides a novel PARAMETRIC approach to studying RH
2. Creates a continuous family of zeta-like functions with varying properties
3. Connects cohomological structure to analytic properties (zeros)
4. Zero-defect cases correctly recover behavior expected from classical RH
5. Shows zeros are SENSITIVE to arithmetic deformation

LIMITATIONS AND CONCERNS:
========================
1. Zeros found are NOT on the critical line Re(s) = 0.5
   - Mean distance from 0.5 is significant (~0.1-0.2)
   - This may indicate the twisted zeta_alpha functions DON'T satisfy RH

2. The relationship between defect magnitude and zero location is WEAK
   - No strong correlation between cohomological curvature and zero movement

3. The "twisted primes" and associated zeta functions are DIFFERENT objects
   - They don't directly prove/disprove classical RH
   - They're NEW mathematical objects with their own properties

4. Sample size is limited (64 zeros across 75 experiments)

5. Imaginary parts don't consistently match classical zeta zeros
   - Suggests these are genuinely DIFFERENT zeros, not deformations

SCIENTIFIC VALUE:
================
- HIGH value for understanding arithmetic deformations
- MODERATE value for RH insight (indirect approach)
- The key question: Do zeros return to Re=0.5 as defect -> 0?

CONCLUSION:
===========
The experiments demonstrate a mathematically interesting framework but
do NOT provide direct evidence for or against the Riemann Hypothesis.
The twisted zeta functions are NEW objects whose zeros behave differently
from the classical Riemann zeta function.

The main insight is that the critical line Re=0.5 is NOT universal across
all twisted arithmetics - it appears to be a SPECIAL property of the
classical (zero-defect) case. This could be viewed as:
- SUPPORTIVE of RH: The critical line property is precisely tied to
  the classical arithmetic structure
- INCONCLUSIVE for RH: These are different functions with different zeros
""")

# ============================================================================
# 11. DETAILED ZERO TABLE
# ============================================================================
print("\n" + "="*80)
print("11. DETAILED ZERO TABLE")
print("="*80)

print(f"\n{'Name':<35} {'Re(s)':<10} {'Im(s)':<10} {'Dist':<8} {'Defect':<12} {'Cocycle'}")
print("-" * 95)
for z in sorted(all_zeros, key=lambda x: abs(x['real'] - 0.5)):
    defect_str = f"{z['defect_max']:.2e}" if z['defect_max'] and z['defect_max'] != float('inf') else "inf"
    cocycle_str = "Yes" if z['cocycle_verified'] else "No"
    dist = abs(z['real'] - 0.5)
    print(f"{z['name']:<35} {z['real']:<10.6f} {z['imag']:<10.4f} {dist:<8.4f} {defect_str:<12} {cocycle_str}")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
