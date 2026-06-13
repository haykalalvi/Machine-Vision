"""
Day 16 - Script 3: Visualize the 2x2 deployment matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

results = np.load('output/matrix_results.npy', allow_pickle=True).item()

precisions = ['FP32', 'Static INT8']
runtimes   = ['ONNX Runtime', 'OpenVINO']

# Build a 2x2 grid of mean latencies
matrix = np.zeros((2, 2))
for i, p in enumerate(precisions):
    for j, r in enumerate(runtimes):
        matrix[i, j] = results[(p, r)].mean()

fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle('Day 16: 2x2 Deployment Matrix — Stage 1 YOLOv8\n'
              'Quantization (Day 15) x Runtime (Day 16)',
              fontsize=14, fontweight='bold')

# --- Heatmap ---
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks([0, 1]); ax.set_xticklabels(runtimes)
ax.set_yticks([0, 1]); ax.set_yticklabels(precisions)
ax.set_title('Latency Heatmap (ms)\nGreen = faster, Red = slower')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{matrix[i,j]:.2f}ms',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white' if matrix[i,j] > matrix.mean() else 'black')
plt.colorbar(im, ax=ax, label='ms')

# --- Grouped bar chart ---
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(precisions))
width = 0.35
for j, runtime in enumerate(runtimes):
    vals = [matrix[i, j] for i in range(2)]
    bars = ax.bar(x + j*width, vals, width, label=runtime,
                   color=['#378ADD', '#1D9E75'][j])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x + width/2)
ax.set_xticklabels(precisions)
ax.set_ylabel('Latency (ms)')
ax.set_title('Side-by-side comparison')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# --- Distribution comparison ---
ax = fig.add_subplot(gs[1, :])
positions = []
data = []
labels = []
colors_list = []
pos = 0
for p in precisions:
    for r in runtimes:
        data.append(results[(p, r)])
        labels.append(f'{p}\n{r}')
        positions.append(pos)
        colors_list.append('#378ADD' if r == 'ONNX Runtime' else '#1D9E75')
        pos += 1

bp = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True,
                widths=0.6)
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Latency distribution across 50 runs (boxplot)\n'
              'Width of box = consistency; outliers = jitter')
ax.set_ylabel('Latency (ms)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/03_deployment_matrix.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: output/03_deployment_matrix.jpg")