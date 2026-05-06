import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

WORKDIR = os.path.dirname(__file__)
CSV = os.path.join(WORKDIR, 'ensemble_probs.csv')
JSON = os.path.join(WORKDIR, 'ensemble_weights_summary.json')
OUT_DIR = WORKDIR

if not os.path.exists(CSV):
    print('ensemble_probs.csv not found; run ensemble_eval.py first')
    raise SystemExit(1)

if not os.path.exists(JSON):
    print('ensemble_weights_summary.json not found; run ensemble_eval.py first')
    raise SystemExit(1)

# load data
df = pd.read_csv(CSV)
labels = df['label'].values
model_cols = [c for c in df.columns if c != 'label']
probs = df[model_cols].values

# split same as ensemble_eval.py
X_train, X_test, y_train, y_test = train_test_split(probs, labels, test_size=0.4, random_state=42, stratify=labels)

# load best validation weights
js = json.load(open(JSON))
best_w = js.get('best_val_weights')
if best_w is None:
    print('No best_val_weights found in JSON')
    raise SystemExit(1)
weights = np.array(best_w)

# compute weighted predictions on validation (X_train)
wp_val = X_train.dot(weights)
wp_val_preds = (wp_val >= 0.5).astype(int)
val_acc = accuracy_score(y_train, wp_val_preds)
print(f'Best validation weighted-avg acc (recomputed): {val_acc:.4f}')

# confusion matrix and report
cm = confusion_matrix(y_train, wp_val_preds)
report = classification_report(y_train, wp_val_preds, target_names=['Cat (0)', 'Dog (1)'], output_dict=True)

# save report
with open(os.path.join(OUT_DIR, 'best_val_weighted_classification_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

# plot confusion matrix (matplotlib)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Predicted Cat', 'Predicted Dog'])
ax.set_yticklabels(['Actual Cat', 'Actual Dog'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='black')
ax.set_title(f'Best-Val Weighted Confusion Matrix\nAcc={val_acc:.4f}')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
fig.colorbar(im, ax=ax)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'BestVal_Weighted_Confusion.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print('Saved', out_path)
plt.close(fig)
