"""
Plot the zeros found by explore_zeros.py
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open('results/explore_zeros_data.json', 'r') as f:
    data = json.load(f)

# Extract zeros
arithmetics = []
real_parts = []
imag_parts = []
colors = []

color_map = {
    'Identity (phi=n)': 'blue',
    'Affine_2n': 'cyan',
    'Affine_n+1': 'green',
    'Exponential_1.5': 'red',
    'Exponential_2.0': 'orange',
    'CoherentTwist_sqrt': 'purple'
}

for result in data['results']:
    arithmetics.append(result['arithmetic'])
    real_parts.append(result['zero']['real'])
    imag_parts.append(result['zero']['imag'])
    colors.append(color_map.get(result['arithmetic'], 'gray'))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the critical line Re(s) = 0.5
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Critical line Re(s)=0.5')

# Plot each zero
for i, (name, re, im, color) in enumerate(zip(arithmetics, real_parts, imag_parts, colors)):
    ax.scatter(re, im, c=color, s=150, marker='o', edgecolors='black', linewidths=1.5,
               label=name, zorder=5)

# Add labels next to points
for name, re, im in zip(arithmetics, real_parts, imag_parts):
    short_name = name.split('_')[0] if '_' in name else name.split(' ')[0]
    ax.annotate(short_name, (re, im), xytext=(8, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

# Formatting
ax.set_xlabel('Re(s)', fontsize=12)
ax.set_ylabel('Im(s)', fontsize=12)
ax.set_title('Zeros of Twisted Zeta Functions $\\zeta_\\phi(s)$\nSearch region: Re$\\in$[0.3, 0.7], Im$\\in$[5, 15]', fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Set axis limits with some padding
ax.set_xlim(0.2, 0.8)
ax.set_ylim(5, 12)

# Add region box for critical strip
ax.axvspan(0, 1, alpha=0.05, color='yellow', label='Critical strip')

plt.tight_layout()
plt.savefig('results/zeros_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('results/zeros_plot.pdf', bbox_inches='tight')
print("Plots saved to results/zeros_plot.png and results/zeros_plot.pdf")
