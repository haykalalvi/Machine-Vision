import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)
import seaborn as sns

# =============================================
# SIMULATE A REALISTIC QC MODEL OUTPUT
# 950 good parts, 50 defective parts
# =============================================

np.random.seed(42)
n_good, n_defect = 950, 50
N = n_good + n_defect

true_labels = np.array([0]*n_good + [1]*n_defect)

# Simulate model confidence scores (0 = definitely good, 1 = definitely defect)
# Good parts: model scores cluster near 0 with some uncertainty
# Defective: model scores cluster near 1 but overlaps with good (hard cases)
scores_good   = np.random.beta(2, 8, n_good)    # peaks near 0.2
scores_defect = np.random.beta(5, 3, n_defect)  # peaks near 0.6
all_scores    = np.concatenate([scores_good, scores_defect])

print("=" * 60)
print("THE METRICS EVERY QC VISION ENGINEER MUST KNOW")
print("=" * 60)

# =============================================
# PART 1: Why accuracy is wrong for QC
# =============================================

threshold    = 0.5
predictions  = (all_scores > threshold).astype(int)
accuracy     = (predictions == true_labels).mean()
defects_caught = predictions[true_labels == 1].sum()
false_alarms   = predictions[true_labels == 0].sum()

print(f"\nAt threshold = {threshold}:")
print(f"  Accuracy:       {accuracy:.2%}  ← looks good!")
print(f"  Defects caught: {defects_caught}/{n_defect} = {defects_caught/n_defect:.0%}")
print(f"  False alarms:   {false_alarms}/{n_good} = {false_alarms/n_good:.0%}")

# =============================================
# PART 2: Confusion matrix — the full picture
# =============================================

cm = confusion_matrix(true_labels, predictions)
TN, FP, FN, TP = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negative  (TN): {TN:3d}  ← good parts correctly passed")
print(f"  False Positive (FP): {FP:3d}  ← good parts wrongly rejected (false alarm)")
print(f"  False Negative (FN): {FN:3d}  ← defective parts wrongly passed ← DANGEROUS")
print(f"  True Positive  (TP): {TP:3d}  ← defective parts correctly caught")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n  Precision: {precision:.3f}  (of flagged defects, how many were real?)")
print(f"  Recall:    {recall:.3f}  (of all real defects, how many did we catch?)")
print(f"  F1 Score:  {f1:.3f}  (harmonic mean of precision and recall)")

print("\n  In manufacturing: RECALL is more critical than precision")
print("  Missing a defect (FN) = product reaches customer = recall")
print("  False alarm (FP) = wasted inspection time = precision")
print("  The trade-off depends on your product and its risk level")

# =============================================
# PART 3: Threshold tuning — precision-recall trade-off
# =============================================

thresholds_to_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
results = []

for t in thresholds_to_test:
    preds  = (all_scores > t).astype(int)
    cm_t   = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm_t.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results.append({
        'threshold': t, 'precision': prec, 'recall': rec, 
        'f1': f1_t, 'fp': fp, 'fn': fn
    })
    print(f"  T={t:.1f} → Recall:{rec:.2f} | Precision:{prec:.2f} | F1:{f1_t:.2f} | FN:{fn} | FP:{fp}")

print("\n  → High recall threshold (T=0.2): catch more defects, more false alarms")
print("  → High precision threshold (T=0.8): fewer false alarms, miss more defects")

# =============================================
# PART 4: ROC Curve and AUROC — threshold-independent
# =============================================

fpr, tpr, roc_thresholds = roc_curve(true_labels, all_scores)
roc_auc = auc(fpr, tpr)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(true_labels, all_scores)
avg_precision = average_precision_score(true_labels, all_scores)

# =============================================
# COMPREHENSIVE VISUALIZATION
# =============================================

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Complete QC Metrics Dashboard\nManufacturing: 950 Good | 50 Defective Parts',
             fontsize=14, fontweight='bold')

# 1. Score distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(scores_good,   bins=30, alpha=0.7, color='#2ca02c', label=f'Good ({n_good})',    density=True)
ax1.hist(scores_defect, bins=30, alpha=0.7, color='#d62728', label=f'Defect ({n_defect})', density=True)
ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
ax1.set_xlabel('Model confidence score')
ax1.set_ylabel('Density')
ax1.set_title('Score Distribution\nOverlap region = hard cases')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Pred: Good', 'Pred: Defect'],
            yticklabels=['True: Good', 'True: Defect'],
            annot_kws={'size': 14})
ax2.set_title(f'Confusion Matrix (T=0.5)\nF1={f1:.3f} | Recall={recall:.3f}')

# 3. ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(fpr, tpr, color='#1D9E75', linewidth=2.5, label=f'ROC curve (AUC={roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random classifier (AUC=0.5)')
ax3.fill_between(fpr, tpr, alpha=0.1, color='#1D9E75')
ax3.set_xlabel('False Positive Rate\n(false alarm rate)')
ax3.set_ylabel('True Positive Rate\n(recall / defect catch rate)')
ax3.set_title(f'ROC Curve\nAUROC={roc_auc:.3f} (higher = better)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(recall_curve, precision_curve, color='#9467bd', linewidth=2.5,
         label=f'PR curve (AP={avg_precision:.3f})')
ax4.fill_between(recall_curve, precision_curve, alpha=0.1, color='#9467bd')
ax4.set_xlabel('Recall (defect catch rate)')
ax4.set_ylabel('Precision (alarm accuracy)')
ax4.set_title('Precision-Recall Curve\nUse when classes are imbalanced')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Threshold tuning table
ax5 = fig.add_subplot(gs[1, 1])
t_vals  = [r['threshold'] for r in results]
rec_v   = [r['recall']    for r in results]
prec_v  = [r['precision'] for r in results]
f1_v    = [r['f1']        for r in results]

ax5.plot(t_vals, rec_v,  'o-', color='#d62728', linewidth=2.5, label='Recall', markersize=8)
ax5.plot(t_vals, prec_v, 's-', color='#2ca02c', linewidth=2.5, label='Precision', markersize=8)
ax5.plot(t_vals, f1_v,   '^-', color='#1D9E75', linewidth=2.5, label='F1', markersize=8)
ax5.axvline(0.3, color='orange', linestyle='--', linewidth=2,
            label='Recommended T=0.3\n(high recall for QC)')
ax5.set_xlabel('Decision threshold')
ax5.set_ylabel('Score')
ax5.set_title('Threshold vs Metrics\nLower T = catch more defects')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. The industry decision table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [
    ['Metric', 'Value', 'Industry meaning'],
    ['AUROC', f'{roc_auc:.3f}', 'Overall separability'],
    ['Recall @ T=0.3', f'{results[1]["recall"]:.3f}', '% defects caught'],
    ['Precision @ T=0.3', f'{results[1]["precision"]:.3f}', '% alarms real'],
    ['F1 @ T=0.3', f'{results[1]["f1"]:.3f}', 'Balance score'],
    ['False negatives', str(results[1]["fn"]), 'Escaped defects'],
    ['False positives', str(results[1]["fp"]), 'Wasted inspections'],
]

table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)
ax6.set_title('Industry Metrics Summary', fontweight='bold', pad=20)

plt.savefig('output/03_complete_metrics_dashboard.jpg', dpi=150, bbox_inches='tight')
plt.show()