"""
Plot results from the full 60+ arithmetic experiment.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open('results/full_experiment/results_latest.json', 'r') as f:
    results = json.load(f)

# Extract zeros by category
categories = ['Hierarchy', 'CoherentTwist', 'Exponential', 'Arbitrary']
colors = {'Hierarchy': 'blue', 'CoherentTwist': 'green', 'Exponential': 'red', 'Arbitrary': 'purple'}
markers = {'Hierarchy': 'o', 'CoherentTwist': 's', 'Exponential': '^', 'Arbitrary': 'd'}

zeros_by_cat = {cat: [] for cat in categories}
all_zeros = []

for r in results:
    if r.get('zeros'):
        for z in r['zeros']:
            zero_data = {
                'name': r['name'],
                'category': r['category'],
                'real': z['real'],
                'imag': z['imag'],
                'residual': z['residual'],
                'converged': z['converged']
            }
            zeros_by_cat[r['category']].append(zero_data)
            all_zeros.append(zero_data)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# ============================================================
# Plot 1: All zeros in complex plane
# ============================================================
ax1 = fig.add_subplot(2, 2, 1)

ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Critical line Re(s)=0.5')

for cat in categories:
    if zeros_by_cat[cat]:
        re = [z['real'] for z in zeros_by_cat[cat]]
        im = [z['imag'] for z in zeros_by_cat[cat]]
        ax1.scatter(re, im, c=colors[cat], marker=markers[cat], s=80,
                   label=f'{cat} ({len(re)})', alpha=0.7, edgecolors='black', linewidths=0.5)

ax1.set_xlabel('Re(s)', fontsize=11)
ax1.set_ylabel('Im(s)', fontsize=11)
ax1.set_title('Zeros of Twisted Zeta Functions by Category', fontsize=12)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.1, 0.8)

# ============================================================
# Plot 2: Histogram of real parts
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)

all_re = [z['real'] for z in all_zeros]
ax2.hist(all_re, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Critical line Re(s)=0.5')
ax2.axvline(x=np.mean(all_re), color='orange', linestyle='-', linewidth=2,
            label=f'Mean = {np.mean(all_re):.3f}')

ax2.set_xlabel('Re(s)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title(f'Distribution of Zero Real Parts (n={len(all_re)})', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ============================================================
# Plot 3: Real part by category (box plot)
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)

data_for_box = []
labels_for_box = []
for cat in categories:
    if zeros_by_cat[cat]:
        data_for_box.append([z['real'] for z in zeros_by_cat[cat]])
        labels_for_box.append(cat)

bp = ax3.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
for patch, cat in zip(bp['boxes'], labels_for_box):
    patch.set_facecolor(colors[cat])
    patch.set_alpha(0.6)

ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Critical line')
ax3.set_ylabel('Re(s)', fontsize=11)
ax3.set_title('Zero Real Parts by Category', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================
# Plot 4: Zeros colored by distance from critical line
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)

re_vals = np.array([z['real'] for z in all_zeros])
im_vals = np.array([z['imag'] for z in all_zeros])
dist_from_critical = np.abs(re_vals - 0.5)

scatter = ax4.scatter(re_vals, im_vals, c=dist_from_critical, cmap='RdYlGn_r',
                      s=100, edgecolors='black', linewidths=0.5)
ax4.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('|Re(s) - 0.5|', fontsize=10)

ax4.set_xlabel('Re(s)', fontsize=11)
ax4.set_ylabel('Im(s)', fontsize=11)
ax4.set_title('Zeros Colored by Distance from Critical Line', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0.1, 0.8)

plt.tight_layout()
plt.savefig('results/full_experiment/zeros_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('results/full_experiment/zeros_analysis.pdf', bbox_inches='tight')
print("Saved: results/full_experiment/zeros_analysis.png")

# ============================================================
# Additional plot: Exponential family α vs Re(s)
# ============================================================
fig2, ax5 = plt.subplots(figsize=(10, 6))

exp_results = [r for r in results if r['category'] == 'Exponential' and r.get('zeros')]
alphas = []
re_parts = []

for r in exp_results:
    # Extract alpha from name
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

    for z in r['zeros']:
        alphas.append(alpha)
        re_parts.append(z['real'])

ax5.scatter(alphas, re_parts, c='red', s=100, edgecolors='black', linewidths=1, zorder=5)
ax5.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Critical line Re(s)=0.5')

# Fit a trend line
if len(alphas) > 2:
    z = np.polyfit(alphas, re_parts, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(alphas), max(alphas), 100)
    ax5.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, linewidth=2, label='Quadratic fit')

ax5.set_xlabel('α (exponential base)', fontsize=12)
ax5.set_ylabel('Re(s) of zero', fontsize=12)
ax5.set_title('Exponential Transfer: Zero Real Part vs α', fontsize=13)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/full_experiment/exponential_zeros.png', dpi=150, bbox_inches='tight')
plt.savefig('results/full_experiment/exponential_zeros.pdf', bbox_inches='tight')
print("Saved: results/full_experiment/exponential_zeros.png")

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "="*60)
print("ZERO STATISTICS")
print("="*60)
print(f"Total zeros found: {len(all_zeros)}")
print(f"Mean Re(s): {np.mean(all_re):.4f}")
print(f"Std Re(s):  {np.std(all_re):.4f}")
print(f"Min Re(s):  {min(all_re):.4f}")
print(f"Max Re(s):  {max(all_re):.4f}")
print(f"\nZeros with Re(s) < 0.5: {sum(1 for r in all_re if r < 0.5)}")
print(f"Zeros with Re(s) > 0.5: {sum(1 for r in all_re if r > 0.5)}")
print(f"Zeros with |Re(s) - 0.5| < 0.05: {sum(1 for r in all_re if abs(r - 0.5) < 0.05)}")

print("\nBy category:")
for cat in categories:
    if zeros_by_cat[cat]:
        re_cat = [z['real'] for z in zeros_by_cat[cat]]
        print(f"  {cat}: n={len(re_cat)}, mean Re={np.mean(re_cat):.4f}, std={np.std(re_cat):.4f}")
