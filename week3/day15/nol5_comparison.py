"""
Day 15 - Script 5: Final Decision Matrix
Combines speed (Scripts 1-3) and accuracy (Script 4) into one
decision: which model version should ship to production?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fp32   = np.load('output/fp32_baseline.npy', allow_pickle=True).item()
dyn    = np.load('output/int8_dynamic.npy', allow_pickle=True).item()
stat   = np.load('output/int8_static.npy', allow_pickle=True).item()

# Fill these in from your Script 4 output
DYNAMIC_IOU = 0.95   # <- replace with your actual mean IoU
STATIC_IOU  = 0.92   # <- replace with your actual mean IoU

fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle('Day 15: Quantization Decision Matrix\nStage 1 YOLOv8 -- MVTec Leather',
             fontsize=14, fontweight='bold')

# --- Speed comparison ---
ax = fig.add_subplot(gs[0, 0])
names  = ['FP32', 'Dynamic\nINT8', 'Static\nINT8']
times  = [fp32['mean_ms'], dyn['mean_ms'], stat['mean_ms']]
colors = ['#888780', '#378ADD', '#1D9E75']
bars = ax.bar(names, times, color=colors)
ax.set_title('Latency (lower is better)')
ax.set_ylabel('ms')
for bar, t in zip(bars, times):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{t:.1f}ms', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# --- FPS comparison with 30 FPS target line ---
ax = fig.add_subplot(gs[0, 1])
fps_vals = [fp32['fps'], dyn['fps'], stat['fps']]
bars = ax.bar(names, fps_vals, color=colors)
ax.axhline(30, color='red', linestyle='--', linewidth=2, label='30 FPS target')
ax.axhline(10, color='orange', linestyle='--', linewidth=1.5, label='10 FPS (Day 14 target)')
ax.set_title('Throughput (higher is better)')
ax.set_ylabel('FPS')
for bar, f in zip(bars, fps_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{f:.1f}', ha='center', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# --- Accuracy retention ---
ax = fig.add_subplot(gs[1, 0])
ious = [1.0, DYNAMIC_IOU, STATIC_IOU]
bar_colors = ['#888780' if i==1.0 else
              ('#1D9E75' if i>0.9 else '#D85A30')
              for i in ious]
bars = ax.bar(names, ious, color=bar_colors)
ax.axhline(0.9, color='black', linestyle='--', label='0.90 acceptable threshold')
ax.set_title('Accuracy retention (Box IoU vs FP32)')
ax.set_ylabel('Mean IoU')
ax.set_ylim(0, 1.05)
for bar, i in zip(bars, ious):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'{i:.3f}', ha='center', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# --- Decision summary table ---
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')

def verdict(speedup, iou):
    if iou < 0.7:
        return 'REJECT - broken'
    elif iou < 0.9:
        return 'CAUTION - verify'
    elif speedup > 1.2:
        return 'RECOMMEND'
    else:
        return 'MARGINAL - no real gain'

table_data = [
    ['Version', 'Latency', 'FPS', 'IoU', 'Verdict'],
    ['FP32', f"{fp32['mean_ms']:.1f}ms", f"{fp32['fps']:.1f}", '1.000', 'Baseline'],
    ['Dynamic INT8', f"{dyn['mean_ms']:.1f}ms", f"{dyn['fps']:.1f}",
     f"{DYNAMIC_IOU:.3f}", verdict(fp32['mean_ms']/dyn['mean_ms'], DYNAMIC_IOU)],
    ['Static INT8', f"{stat['mean_ms']:.1f}ms", f"{stat['fps']:.1f}",
     f"{STATIC_IOU:.3f}", verdict(fp32['mean_ms']/stat['mean_ms'], STATIC_IOU)],
]
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 2.2)
ax.set_title('Decision Matrix', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('output/05_decision_matrix.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: output/05_decision_matrix.jpg")

print(f"\n{'='*55}")
print("RECOMMENDATION")
print('='*55)
best = max(
    [('Dynamic INT8', dyn['fps'], DYNAMIC_IOU),
     ('Static INT8', stat['fps'], STATIC_IOU)],
    key=lambda x: x[1] if x[2] > 0.9 else -1
)
print(f"Recommended: {best[0]}")
print(f"  FPS: {best[1]:.1f} (vs FP32: {fp32['fps']:.1f})")
print(f"  Accuracy retained: {best[2]:.1%}")