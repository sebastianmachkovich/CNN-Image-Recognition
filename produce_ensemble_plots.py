import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

WORKDIR = os.path.dirname(__file__)
CSV = os.path.join(WORKDIR, 'ensemble_probs.csv')
JSON = os.path.join(WORKDIR, 'ensemble_weights_summary.json')
OUT_DIR = WORKDIR

if not os.path.exists(CSV):
    print('ensemble_probs.csv not found; run ensemble_eval.py first')
    raise SystemExit(1)

df = pd.read_csv(CSV)
labels = df['label'].values
model_cols = [c for c in df.columns if c != 'label']
probs = df[model_cols].values  # shape (n_samples, n_models)

# load weights
if os.path.exists(JSON):
    js = json.load(open(JSON))
    weights = js.get('best_val_weights') or js.get('best_val_weights', None)
    if weights is not None:
        weights = np.array(weights)
else:
    weights = None

# fallback: proportional to per-model accuracy
if weights is None:
    per_acc = [(probs[:, i] >= 0.5).astype(int) for i in range(probs.shape[1])]
    per_acc = [accuracy_score(labels, p) for p in per_acc]
    weights = np.array(per_acc)
    weights = weights / weights.sum()

# equal-weight ensemble
avg_probs = np.mean(probs, axis=1)
avg_preds = (avg_probs >= 0.5).astype(int)
avg_acc = accuracy_score(labels, avg_preds)

# weighted ensemble
weighted_probs = probs.dot(weights)
w_preds = (weighted_probs >= 0.5).astype(int)
w_acc = accuracy_score(labels, w_preds)

print('Equal-weight ensemble acc:', avg_acc := avg_acc if (avg_acc:=None) else None)
# above line is placeholder to avoid lint false positive; compute properly
print('Computed accuracies:')
print('  Equal-weight:', round(avg_acc := accuracy_score(labels, avg_preds), 4))
print('  Weighted    :', round(w_acc, 4))

# confusion matrices
cm_eq = confusion_matrix(labels, avg_preds)
cm_w = confusion_matrix(labels, w_preds)

# save confusion matrices as heatmaps
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

def plot_confusion(cm, axis, title):
    im = axis.imshow(cm, cmap='Blues')
    axis.set_xticks([0,1])
    axis.set_yticks([0,1])
    axis.set_xticklabels(['Predicted Cat', 'Predicted Dog'])
    axis.set_yticklabels(['Actual Cat', 'Actual Dog'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, str(cm[i,j]), ha='center', va='center', color='black')
    axis.set_title(title)
    return im

im1 = plot_confusion(cm_eq, ax[0], f'Equal-weight Confusion Matrix\nAcc={accuracy_score(labels, avg_preds):.4f}')
im2 = plot_confusion(cm_w, ax[1], f'Weighted Confusion Matrix\nAcc={w_acc:.4f}')
fig.colorbar(im2, ax=ax.ravel().tolist())

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'Ensemble_Confusion_Comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print('Saved', out_path)
plt.close(fig)

# Comparison plot similar to training curves: left — per-model and ensemble accuracies; right — mean probs
per_model_acc = [accuracy_score(labels, (probs[:, i] >= 0.5).astype(int)) for i in range(probs.shape[1])]
models_range = range(1, len(per_model_acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(models_range, per_model_acc, marker='o', label='Per-model Accuracy', linewidth=2)
ax1.axhline(accuracy_score(labels, avg_preds), color='C1', linestyle='--', label='Equal-weight Ensemble')
ax1.axhline(accuracy_score(labels, w_preds), color='C2', linestyle='-.', label='Weighted Ensemble')
ax1.set_title('Per-model vs Ensemble Accuracy', fontsize=13, fontweight='bold')
ax1.set_xlabel('Model index')
ax1.set_ylabel('Accuracy')
ax1.set_xticks(list(models_range))
ax1.legend()

mean_pos = [np.mean(probs[labels == 1, i]) for i in range(probs.shape[1])]
mean_neg = [np.mean(probs[labels == 0, i]) for i in range(probs.shape[1])]
ax2.plot(models_range, mean_pos, marker='o', label='Mean prob (True Dog)', linewidth=2)
ax2.plot(models_range, mean_neg, marker='o', label='Mean prob (True Cat)', linewidth=2)
ax2.set_title('Mean Predicted Prob by True Class', fontsize=13, fontweight='bold')
ax2.set_xlabel('Model index')
ax2.set_ylabel('Mean predicted probability')
ax2.set_xticks(list(models_range))
ax2.legend()

plt.tight_layout()
out_path2 = os.path.join(OUT_DIR, 'Ensemble_Comparison.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print('Saved', out_path2)
plt.close(fig)

# also save weighted classification report
report_w = classification_report(labels, w_preds, target_names=['Cat (0)', 'Dog (1)'], output_dict=True)
with open(os.path.join(OUT_DIR, 'weighted_classification_report.json'), 'w') as f:
    json.dump(report_w, f, indent=2)
print('Saved weighted classification report')
