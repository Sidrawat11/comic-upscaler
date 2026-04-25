import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("cache/rtx-4060-laptop-gpu.json") as f:
    data = json.load(f)

valid = [e for e in data["entries"] if e["valid"]]
pixels = np.array([e["width"] * e["height"] for e in valid])
vram = np.array([e["peak_vram_mb"] for e in valid])
labels = [f'{e["width"]}x{e["height"]}' for e in valid]

# Fit line
slope, intercept = np.polyfit(pixels, vram, 1)
fit_line = slope * pixels + intercept
r_squared = 1 - np.sum((vram - fit_line)**2) / np.sum((vram - np.mean(vram))**2)

# Color by height
heights = [e["height"] for e in valid]
unique_heights = sorted(set(heights))
colors = {128: '#2ac3de', 256: '#7aa2f7', 512: '#bb9af7', 1024: '#f7768e'}

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#1a1b26')
ax.set_facecolor('#1f2335')

# Plot points colored by height
for h in unique_heights:
    mask = [i for i, e in enumerate(valid) if e["height"] == h]
    px = pixels[mask]
    vr = vram[mask]
    color = colors.get(h, '#9ece6a')
    ax.scatter(px, vr, c=color, s=60, label=f'{h}px height', zorder=3, edgecolors='white', linewidth=0.5)

# Labels
for i, label in enumerate(labels):
    ax.annotate(label, (pixels[i], vram[i]), fontsize=6, color='#c0caf5',
                xytext=(5, 5), textcoords='offset points')

# Fit line
sort_idx = np.argsort(pixels)
ax.plot(pixels[sort_idx], fit_line[sort_idx], '--', color='#e0af68', linewidth=2,
        label=f'Fit: {slope:.4f} × pixels + {intercept:.1f}')

# VRAM budget lines
total_vram = data["available_vram"]
ax.axhline(y=total_vram * 0.85, color='#f7768e', linestyle=':', linewidth=1.5, label=f'85% budget ({total_vram*0.85:.0f} MB)')
ax.axhline(y=total_vram * 0.75, color='#ff9e64', linestyle=':', linewidth=1.5, label=f'75% budget ({total_vram*0.75:.0f} MB)')

# Equation box
eq_text = f'VRAM = {slope:.4f} × (W×H) + {intercept:.1f} MB\nR² = {r_squared:.4f}\nSlope = {slope*1000:.2f} MB per 1000 pixels'
ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=10, color='#9ece6a',
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#24283b', edgecolor='#3b4261'))

ax.set_xlabel('Total Pixels (W × H)', color='#c0caf5', fontsize=12)
ax.set_ylabel('Peak VRAM (MB)', color='#c0caf5', fontsize=12)
ax.set_title('RRDBNet-23 FP16 VRAM vs Pixel Count — RTX 4060 Laptop', color='#7aa2f7', fontsize=14)
ax.tick_params(colors='#c0caf5')
ax.grid(True, color='#3b4261', alpha=0.5)
ax.legend(loc='lower right', fontsize=9, facecolor='#24283b', edgecolor='#3b4261', labelcolor='#c0caf5')

for spine in ax.spines.values():
    spine.set_color('#3b4261')

plt.tight_layout()
plt.savefig("vram_vs_pixels_detailed.png", dpi=200, facecolor='#1a1b26')
print(f"\nslope: {slope:.4f} MB/pixel")
print(f"intercept: {intercept:.1f} MB")
print(f"R²: {r_squared:.4f}")
print(f"Formula: estimated_vram_mb = {slope:.4f} * (h * w) + {intercept:.1f}")
print(f"\nSaved vram_vs_pixels_detailed.png")