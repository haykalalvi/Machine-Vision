"""
Day 13 - Script 3: Final Comparison — U-Net Segmentation vs PatchCore
This is what you present in interviews and your portfolio.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load saved results
pc_scores = np.load('output/patchcore_scores.npy')
pc_labels = np.load('output/patchcore_labels.npy')

from sklearn.metrics import roc_auc_score
pc_auroc = roc_auc_score(pc_labels, pc_scores)

# These come from your Day 12 results — fill in from your notes.md
unet_iou  = 0.55   # replace with your actual Day 12 val IoU
unet_dice = 0.68   # replace with your actual Day 12 val Dice

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    'Method Comparison: Supervised U-Net vs Unsupervised PatchCore\n'
    'MVTec AD — Leather Category',
    fontsize=14, fontweight='bold'
)

# Comparison table
ax = fig.add_subplot(gs[0, 0])
ax.axis('off')
comparison = [
    ['Criterion',          'U-Net (Day 12)',    'PatchCore (Day 13)'],
    ['Labels needed',      'Pixel masks ✗',    'None ✓'],
    ['Training time',      '~30 min',          '<2 min'],
    ['New defect types',   'Retrain needed ✗', 'Auto-detected ✓'],
    ['Output',             'Pixel mask',       'Anomaly score + map'],
    ['Val IoU / AUROC',    f'{unet_iou:.3f}',  f'{pc_auroc:.3f}'],
    ['Industry adoption',  'Niche',            'Dominant ✓'],
]
table = ax.table(
    cellText=comparison[1:], colLabels=comparison[0],
    loc='center', cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 2.2)
ax.set_title('Method Comparison', fontweight='bold', pad=20)

# Score distribution for PatchCore
ax = fig.add_subplot(gs[0, 1])
good_s   = pc_scores[pc_labels == 0]
defect_s = pc_scores[pc_labels == 1]
ax.hist(good_s,   bins=20, alpha=0.7, color='#1D9E75', label=f'Good ({len(good_s)})',    density=True)
ax.hist(defect_s, bins=20, alpha=0.7, color='#D85A30', label=f'Defect ({len(defect_s)})', density=True)
ax.set_title(f'PatchCore Score Distribution\nAUROC = {pc_auroc:.4f}')
ax.set_xlabel('Anomaly score'); ax.set_ylabel('Density')
ax.legend(); ax.grid(True, alpha=0.3)

# Threshold sweep for PatchCore
ax       = fig.add_subplot(gs[0, 2])
thresh   = np.linspace(0, 1, 100)
recalls, precisions, f1s = [], [], []
for t in thresh:
    preds  = (pc_scores > t).astype(int)
    tp = ((preds==1) & (pc_labels==1)).sum()
    fp = ((preds==1) & (pc_labels==0)).sum()
    fn = ((preds==0) & (pc_labels==1)).sum()
    r  = tp/(tp+fn) if (tp+fn)>0 else 0
    p  = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    recalls.append(r); precisions.append(p); f1s.append(f1)

best_t = thresh[np.argmax(f1s)]
ax.plot(thresh, recalls,    label='Recall',    color='#D85A30', linewidth=2)
ax.plot(thresh, precisions, label='Precision', color='#378ADD', linewidth=2)
ax.plot(thresh, f1s,        label='F1',        color='#1D9E75', linewidth=2.5)
ax.axvline(best_t, color='black', linestyle='--',
           label=f'Best F1 threshold={best_t:.2f}')
ax.set_title('PatchCore Threshold Analysis\nFor Manufacturing: choose recall > 0.90')
ax.set_xlabel('Threshold'); ax.legend(); ax.grid(True, alpha=0.3)

# When to use which method
ax = fig.add_subplot(gs[1, :])
ax.axis('off')
decision_text = """
WHEN TO USE EACH METHOD IN INDUSTRY — The Decision Framework

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  USE PATCHCORE (Anomaly Detection) when:                  │  USE U-NET (Supervised Segmentation) when:      │
│  • You have < 50 defect images per class                  │  • You have 200+ labeled defect images           │
│  • New defect types may appear in production             │  • Defect types are well-defined and stable       │
│  • You cannot afford annotation cost                     │  • Precise pixel boundary is required             │
│  • You need to deploy in days, not weeks                 │  • Downstream robot needs exact defect shape      │
│  • You want zero-shot generalization                     │  • Regulatory compliance requires explainability  │
│                                                           │                                                   │
│  Industry examples: scratch detection, contamination,    │  Industry examples: medical imaging, precise      │
│  any "golden sample" quality system                      │  measurement, surgical robotics, wafer inspection │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
ax.text(0.01, 0.5, decision_text, transform=ax.transAxes,
        fontsize=10, va='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('output/03_method_comparison.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: output/03_method_comparison.jpg")
print("\n✓ Day 13 complete!")
print("  Commit: git add . && git commit -m 'day-13: PatchCore anomaly detection — AUROC=X.XX'")