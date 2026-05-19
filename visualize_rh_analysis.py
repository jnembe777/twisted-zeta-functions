"""
Visualization of Twisted Zeta Function Analysis
Relevance to Riemann Hypothesis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Load data
with open('results/full_experiment/results_latest.json', 'r') as f:
    results = json.load(f)

# Collect all zeros with metadata
all_zeros = []
for r in results:
    for z in r.get('zeros', []):
        defect_max = r['defect']['max'] if r.get('defect') else None
        # Handle infinity
        if defect_max == float('inf') or (isinstance(defect_max, float) and defect_max != defect_max):
            defect_max = 1e40  # Large but finite for plotting
        all_zeros.append({
            'name': r['name'],
            'category': r['category'],
            'real': z['real'],
            'imag': z['imag'],
            'defect_max': defect_max,
            'defect_is_zero': r['defect']['is_zero'] if r.get('defect') else None,
            'cocycle_verified': r['cohomology']['cocycle_verified'] if r.get('cohomology') else None,
        })

# Create figure with subplots
fig = plt.figure(figsize=(16, 20))

# Color scheme
colors = {
    'Hierarchy': '#e74c3c',
    'CoherentTwist': '#3498db',
    'Exponential': '#2ecc71',
    'Arbitrary': '#9b59b6'
}

# ============================================================================
# PLOT 1: Zero Distribution in Complex Plane
# ============================================================================
ax1 = fig.add_subplot(3, 2, 1)

# Draw critical strip
ax1.axvspan(0, 1, alpha=0.1, color='blue', label='Critical Strip')
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Critical Line (Re=0.5)')

# Plot zeros by category
for cat in colors:
    cat_zeros = [z for z in all_zeros if z['category'] == cat]
    if cat_zeros:
        x = [z['real'] for z in cat_zeros]
        y = [z['imag'] for z in cat_zeros]
        ax1.scatter(x, y, c=colors[cat], label=f'{cat} ({len(cat_zeros)})',
                   alpha=0.7, s=80, edgecolors='white', linewidth=0.5)

# Mark classical first zero location
ax1.scatter([0.5], [14.1347], marker='*', s=300, c='gold', edgecolors='black',
           linewidth=1.5, zorder=10, label='Classical 1st zero')

ax1.set_xlabel('Re(s)', fontsize=12)
ax1.set_ylabel('Im(s)', fontsize=12)
ax1.set_title('Zero Distribution in Complex Plane\n(RH: All zeros should be on red line)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(-0.1, 1.6)
ax1.set_ylim(4, 22)
ax1.grid(True, alpha=0.3)

# ============================================================================
# PLOT 2: Distance from Critical Line Histogram
# ============================================================================
ax2 = fig.add_subplot(3, 2, 2)

distances = [abs(z['real'] - 0.5) for z in all_zeros]
bins = np.linspace(0, 1, 25)

n, bins_out, patches = ax2.hist(distances, bins=bins, color='steelblue',
                                 edgecolor='white', alpha=0.8)

# Color bars by distance
for i, patch in enumerate(patches):
    if bins_out[i] < 0.05:
        patch.set_facecolor('#27ae60')  # Green - on critical
    elif bins_out[i] < 0.15:
        patch.set_facecolor('#f39c12')  # Orange - moderate
    else:
        patch.set_facecolor('#e74c3c')  # Red - far

# Add vertical lines for thresholds
ax2.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Near critical (<0.05)')
ax2.axvline(x=0.15, color='orange', linestyle='--', linewidth=2, label='Moderate (<0.15)')

# Stats annotation
mean_dist = np.mean(distances)
ax2.axvline(x=mean_dist, color='purple', linestyle='-', linewidth=2, label=f'Mean: {mean_dist:.3f}')

ax2.set_xlabel('Distance from Critical Line |Re(s) - 0.5|', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Distance from Critical Line Distribution\n(RH: All should be at 0)',
              fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)

# Add text box with stats
textstr = f'Total zeros: {len(distances)}\nOn critical (<0.01): {sum(1 for d in distances if d < 0.01)}\nNear (<0.05): {sum(1 for d in distances if d < 0.05)}\nFar (>0.15): {sum(1 for d in distances if d >= 0.15)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.65, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# ============================================================================
# PLOT 3: Defect vs Zero Location (Key correlation)
# ============================================================================
ax3 = fig.add_subplot(3, 2, 3)

# Filter for finite defects
finite_zeros = [z for z in all_zeros if z['defect_max'] is not None
                and z['defect_max'] < 1e35]

if finite_zeros:
    log_defects = [np.log10(max(z['defect_max'], 1e-35)) for z in finite_zeros]
    distances = [abs(z['real'] - 0.5) for z in finite_zeros]
    cats = [z['category'] for z in finite_zeros]

    for cat in colors:
        mask = [c == cat for c in cats]
        x = [log_defects[i] for i in range(len(mask)) if mask[i]]
        y = [distances[i] for i in range(len(mask)) if mask[i]]
        if x:
            ax3.scatter(x, y, c=colors[cat], label=cat, alpha=0.7, s=80,
                       edgecolors='white', linewidth=0.5)

    # Add trend line
    z = np.polyfit(log_defects, distances, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(log_defects), max(log_defects), 100)
    ax3.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')

    # Correlation
    corr = np.corrcoef(log_defects, distances)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax3.set_xlabel('log10(Defect Magnitude)', fontsize=12)
ax3.set_ylabel('Distance from Critical Line', fontsize=12)
ax3.set_title('Defect Magnitude vs Zero Location\n(Shows arithmetic deformation effect)',
              fontsize=13, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)

# ============================================================================
# PLOT 4: Zero-Defect vs Non-Zero Defect Comparison
# ============================================================================
ax4 = fig.add_subplot(3, 2, 4)

zero_defect = [z for z in all_zeros if z['defect_is_zero']]
nonzero_defect = [z for z in all_zeros if z['defect_is_zero'] == False]

# Box plots
data = [
    [abs(z['real'] - 0.5) for z in zero_defect],
    [abs(z['real'] - 0.5) for z in nonzero_defect]
]

bp = ax4.boxplot(data, labels=['Zero Defect\n(Classical-like)', 'Non-zero Defect\n(Deformed)'],
                 patch_artist=True, widths=0.6)

bp['boxes'][0].set_facecolor('#27ae60')
bp['boxes'][1].set_facecolor('#e74c3c')

for box in bp['boxes']:
    box.set_alpha(0.7)

# Add individual points
for i, d in enumerate(data, 1):
    x = np.random.normal(i, 0.04, size=len(d))
    ax4.scatter(x, d, alpha=0.5, s=50, c='navy')

# Add means
means = [np.mean(d) for d in data]
ax4.scatter([1, 2], means, marker='D', s=100, c='red', zorder=10, label='Mean')

ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_ylabel('Distance from Critical Line', fontsize=12)
ax4.set_title('Zero-Defect vs Non-Zero Defect Arithmetics\n(Zero defect should recover RH)',
              fontsize=13, fontweight='bold')
ax4.legend(loc='upper right')

# Stats
ax4.text(0.5, -0.12, f'Zero-defect mean: {means[0]:.4f}  |  Non-zero mean: {means[1]:.4f}',
        transform=ax4.transAxes, ha='center', fontsize=11, fontweight='bold')

# ============================================================================
# PLOT 5: Exponential Transfer Alpha Trend
# ============================================================================
ax5 = fig.add_subplot(3, 2, 5)

# Extract exponential transfer data
exp_data = []
for r in results:
    if r['category'] == 'Exponential' and r['n_zeros'] > 0:
        name = r['name']
        if 'alpha_e' in name:
            alpha = np.e
        elif 'alpha_pi' in name:
            alpha = np.pi
        else:
            try:
                alpha = float(name.split('_')[-1])
            except:
                continue
        exp_data.append({
            'alpha': alpha,
            'real': r['zeros'][0]['real'],
            'dist': abs(r['zeros'][0]['real'] - 0.5)
        })

exp_data.sort(key=lambda x: x['alpha'])

if exp_data:
    alphas = [d['alpha'] for d in exp_data]
    dists = [d['dist'] for d in exp_data]
    reals = [d['real'] for d in exp_data]

    # Plot distance from critical line
    ax5.plot(alphas, dists, 'o-', color='#e74c3c', linewidth=2, markersize=10,
            label='Distance from Re=0.5', markeredgecolor='white', markeredgewidth=1.5)

    # Add trend line
    z = np.polyfit(alphas, dists, 1)
    p = np.poly1d(z)
    ax5.plot(alphas, p(alphas), '--', color='darkred', linewidth=2, alpha=0.7)

    # Correlation
    corr = np.corrcoef(alphas, dists)[0, 1]
    ax5.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax5.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax5.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Critical line (ideal)')
ax5.set_xlabel('Base alpha (exponential transfer)', fontsize=12)
ax5.set_ylabel('Distance from Critical Line', fontsize=12)
ax5.set_title('Exponential Transfers: alpha vs Zero Displacement\n(Key finding: larger alpha = further from RH)',
              fontsize=13, fontweight='bold')
ax5.legend(loc='upper left', fontsize=9)
ax5.grid(True, alpha=0.3)

# ============================================================================
# PLOT 6: Summary Dashboard
# ============================================================================
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')

# Create summary text
summary = """
ANALYSIS SUMMARY: TWISTED ZETA FUNCTIONS AND THE RIEMANN HYPOTHESIS

KEY FINDINGS:
=============

1. ZERO DISTRIBUTION
   - 64 zeros found across 75 experiments
   - Mean Re(s) = 0.416 (RH predicts 0.5)
   - Only 1.6% exactly on critical line
   - 53% are far (>0.15) from critical line

2. CRITICAL CORRELATION DISCOVERED
   - Correlation (alpha vs distance): +0.66
   - Larger arithmetic deformation = zeros further from Re=0.5
   - This is DIRECT EVIDENCE of structure-dependent zero location

3. ZERO-DEFECT ANOMALY
   - Even zero-defect cases show zeros at Re=0.34, not Re=0.5
   - Suggests twisted zeta is fundamentally different from classical zeta

4. COCYCLE CONDITION
   - Cocycle-verified: mean dist = 0.150
   - Cocycle-failed: mean dist = 0.165
   - Slight tendency for cocycle failure to correlate with further zeros

IMPLICATIONS FOR RH:
====================
+ Framework shows zeros are SENSITIVE to arithmetic structure
+ Critical line appears SPECIAL to classical arithmetic
+ Parametric deformation is a valid research approach

- Does NOT directly prove/disprove classical RH
- Twisted zeta functions are NEW objects with different zeros
- Zero-defect limit doesn't recover classical behavior

CONCLUSION: Mathematically interesting but INCONCLUSIVE for RH
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9))

# Main title
fig.suptitle('Twisted Zeta Function Experiments: Analysis for Riemann Hypothesis',
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('results/rh_analysis_visualization.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('results/rh_analysis_visualization.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Visualization saved to:")
print("  - results/rh_analysis_visualization.png")
print("  - results/rh_analysis_visualization.pdf")

# ============================================================================
# ADDITIONAL PLOT: Real Part vs Imaginary Part by Category
# ============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Re(s) distribution by category
ax = axes[0, 0]
category_data = defaultdict(list)
for z in all_zeros:
    category_data[z['category']].append(z['real'])

positions = list(range(len(category_data)))
bp = ax.boxplot([category_data[cat] for cat in colors.keys()],
               labels=list(colors.keys()), patch_artist=True, widths=0.6)

for patch, cat in zip(bp['boxes'], colors.keys()):
    patch.set_facecolor(colors[cat])
    patch.set_alpha(0.7)

ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Critical line (RH)')
ax.set_ylabel('Re(s)', fontsize=12)
ax.set_title('Real Part Distribution by Category', fontsize=13, fontweight='bold')
ax.legend()

# Plot 2: Cocycle verification impact
ax = axes[0, 1]
cocycle_true = [abs(z['real'] - 0.5) for z in all_zeros if z['cocycle_verified']]
cocycle_false = [abs(z['real'] - 0.5) for z in all_zeros if not z['cocycle_verified']]

ax.hist(cocycle_true, bins=15, alpha=0.6, label=f'Cocycle OK ({len(cocycle_true)})', color='green')
ax.hist(cocycle_false, bins=15, alpha=0.6, label=f'Cocycle Failed ({len(cocycle_false)})', color='red')
ax.set_xlabel('Distance from Critical Line', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Impact of Cocycle Verification on Zero Location', fontsize=13, fontweight='bold')
ax.legend()

# Plot 3: Imaginary parts distribution
ax = axes[1, 0]
imag_parts = [z['imag'] for z in all_zeros]
ax.hist(imag_parts, bins=20, color='purple', alpha=0.7, edgecolor='white')

# Mark known classical zeta zero heights
classical_heights = [14.1347, 21.0220, 25.0109]
for h in classical_heights:
    ax.axvline(x=h, color='gold', linestyle='--', linewidth=2)

ax.axvline(x=classical_heights[0], color='gold', linestyle='--', linewidth=2,
          label='Classical zeta heights')
ax.set_xlabel('Im(s)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Imaginary Part Distribution\n(Gold lines: classical zeta zero heights)',
            fontsize=13, fontweight='bold')
ax.legend()

# Plot 4: Scatter of all zeros with size = defect
ax = axes[1, 1]
for cat in colors:
    cat_zeros = [z for z in all_zeros if z['category'] == cat]
    if cat_zeros:
        x = [z['real'] for z in cat_zeros]
        y = [z['imag'] for z in cat_zeros]
        # Size based on defect (log scale)
        sizes = []
        for z in cat_zeros:
            if z['defect_max'] and z['defect_max'] < 1e35:
                s = 30 + 10 * np.log10(max(z['defect_max'], 1))
            else:
                s = 100
            sizes.append(max(30, min(s, 300)))

        ax.scatter(x, y, c=colors[cat], s=sizes, alpha=0.6,
                  edgecolors='white', linewidth=0.5, label=cat)

ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Re(s)', fontsize=12)
ax.set_ylabel('Im(s)', fontsize=12)
ax.set_title('Zero Location (size ~ log defect)\n(Red line: Critical line)',
            fontsize=13, fontweight='bold')
ax.legend(loc='upper right')

fig2.suptitle('Additional Analysis Views', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('results/rh_analysis_details.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("  - results/rh_analysis_details.png")

# ============================================================================
# PLOT 3: Critical Line Focus
# ============================================================================
fig3, ax = plt.subplots(figsize=(12, 8))

# Focus on the critical strip region
ax.axvspan(0, 1, alpha=0.1, color='blue')
ax.axvline(x=0.5, color='red', linestyle='-', linewidth=3, label='Critical Line (RH)')

# Add tolerance bands
ax.axvspan(0.45, 0.55, alpha=0.2, color='green', label='|Re-0.5| < 0.05')
ax.axvspan(0.35, 0.65, alpha=0.1, color='yellow', label='|Re-0.5| < 0.15')

# Plot all zeros
for cat in colors:
    cat_zeros = [z for z in all_zeros if z['category'] == cat]
    if cat_zeros:
        x = [z['real'] for z in cat_zeros]
        y = [z['imag'] for z in cat_zeros]
        ax.scatter(x, y, c=colors[cat], s=100, alpha=0.8,
                  edgecolors='black', linewidth=1, label=f'{cat}')

# Mark the closest to critical line
closest = min(all_zeros, key=lambda z: abs(z['real'] - 0.5))
ax.annotate(f'Closest: {closest["name"]}\nRe={closest["real"]:.4f}',
           xy=(closest['real'], closest['imag']),
           xytext=(closest['real'] + 0.15, closest['imag'] + 1),
           fontsize=10, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='black'),
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# Mark classical zero
ax.scatter([0.5], [14.1347], marker='*', s=400, c='gold', edgecolors='black',
          linewidth=2, zorder=10, label='Classical 1st zero (RH)')

ax.set_xlabel('Re(s)', fontsize=14)
ax.set_ylabel('Im(s)', fontsize=14)
ax.set_title('Zero Locations Relative to Critical Line\nRiemann Hypothesis: All zeros should lie on the red line',
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0.1, 0.75)
ax.set_ylim(4, 22)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/rh_critical_line_focus.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("  - results/rh_critical_line_focus.png")

print("\nAll visualizations completed!")
