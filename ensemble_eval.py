import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Config
WORKDIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(WORKDIR, "ensemble_models")
TEST_DIR = os.path.join(WORKDIR, "test_set")
BATCH_SIZE = 32


def list_image_paths_and_labels(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    paths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            # skip hidden or helper files and non-image files
            if fname.startswith(".") or fname.startswith("_"):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
                continue
            p = os.path.join(cls_dir, fname)
            if os.path.isfile(p):
                paths.append(p)
                labels.append(idx)
    return paths, np.array(labels), classes


def load_and_preprocess(path, target_size):
    img = Image.open(path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def get_model_input_size(model):
    shape = None
    try:
        shape = model.input_shape
    except Exception:
        try:
            shape = model.layers[0].input_shape
        except Exception:
            pass
    if shape is None:
        return (150, 150, 3)
    # shape is like (None, H, W, C) or (None, C, H, W)
    if len(shape) == 4:
        if shape[1] is None:
            return (shape[2], shape[3], shape[4]) if len(shape) > 4 else (150,150,3)
        if shape[1] in (1,3) and shape[2] and shape[3]:
            # (None, C, H, W) ?
            if shape[1] in (1,3):
                return (shape[2], shape[3], shape[1])
        # assume (None, H, W, C)
        return (shape[1], shape[2], shape[3])
    elif len(shape) == 3:  # (H, W, C)
        return (shape[0], shape[1], shape[2])
    return (150, 150, 3)


def predict_probs_for_model(model, paths, input_size):
    n = len(paths)
    probs = []
    for i in range(0, n, BATCH_SIZE):
        batch_paths = paths[i:i+BATCH_SIZE]
        batch = np.stack([load_and_preprocess(p, input_size) for p in batch_paths], axis=0)
        preds = model.predict(batch, verbose=0)
        preds = np.array(preds)
        # Normalize to single probability of positive class
        if preds.ndim == 2 and preds.shape[1] == 2:
            p = preds[:, 1]
        elif preds.ndim == 2 and preds.shape[1] == 1:
            p = preds[:, 0]
        elif preds.ndim == 1:
            p = preds
        else:
            # fallback: take softmax across last dim
            ex = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            p = ex[:, 1] / np.sum(ex, axis=1)
        # clip
        p = np.clip(p, 0.0, 1.0)
        probs.append(p)
    return np.concatenate(probs, axis=0)


def main():
    print("Scanning test images...")
    paths, labels, classes = list_image_paths_and_labels(TEST_DIR)
    if len(paths) == 0:
        print("No images found in test_set. Check TEST_DIR path.")
        return
    print(f"Found {len(paths)} images across classes: {classes}")

    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.h5")))
    if not model_files:
        print("No models found in ensemble_models/")
        return

    all_probs = []
    model_names = []
    per_model_acc = []

    for mf in model_files:
        print(f"Loading model {mf}...")
        try:
            m = load_model(mf, compile=False)
        except Exception as e:
            print(f"Failed loading {mf}: {e}")
            continue
        input_size = get_model_input_size(m)
        # input_size may be (H,W,C)
        if len(input_size) == 3:
            h, w, c = input_size
        else:
            h, w = input_size[0], input_size[1]
        print(f"Model expects input approx {h}x{w}")
        probs = predict_probs_for_model(m, paths, (h, w, 3))
        # Determine binary preds
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        print(f"Model {os.path.basename(mf)} accuracy: {acc:.4f}")
        all_probs.append(probs)
        model_names.append(os.path.basename(mf))
        per_model_acc.append(acc)

    all_probs = np.vstack(all_probs).T  # shape (n_samples, n_models)
    n_models = all_probs.shape[1]
    print(f"Collected probabilities shape: {all_probs.shape}")

    # Pairwise correlation
    corr = np.corrcoef(all_probs.T)
    print("Pairwise Pearson correlation matrix between model probabilities:")
    print(corr)

    # Pairwise disagreement (fraction of differing binary predictions)
    bin_preds = (all_probs >= 0.5).astype(int)
    disagreement = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            disagreement[i, j] = np.mean(bin_preds[:, i] != bin_preds[:, j])
    print("Pairwise disagreement matrix (fraction where predictions differ):")
    print(disagreement)

    # Equal-weight soft voting
    avg_probs = np.mean(all_probs, axis=1)
    ensemble_preds = (avg_probs >= 0.5).astype(int)
    ensemble_acc = accuracy_score(labels, ensemble_preds)
    print(f"Equal-weight soft-voting ensemble accuracy: {ensemble_acc:.4f}")

    # Train logistic regression stacking (use train/test split on available test_set)
    X_train, X_test, y_train, y_test = train_test_split(all_probs, labels, test_size=0.4, random_state=42, stratify=labels)
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_train, y_train)
    meta_preds = meta.predict(X_test)
    meta_acc = accuracy_score(y_test, meta_preds)
    print(f"Stacking (LogisticRegression) accuracy on split test: {meta_acc:.4f}")

    # Weighted averaging: find weights by random search on validation (X_train)
    def random_search_weights(X_val, y_val, n_iter=2000, seed=0):
        rng = np.random.RandomState(seed)
        best = {'acc': -1, 'w': None}
        M = X_val.shape[1]
        for _ in range(n_iter):
            # sample from Dirichlet to get non-negative weights summing to 1
            w = rng.dirichlet(alpha=np.ones(M))
            wp = X_val.dot(w)
            preds = (wp >= 0.5).astype(int)
            acc = accuracy_score(y_val, preds)
            if acc > best['acc']:
                best['acc'] = acc
                best['w'] = w
        return best

    print("Searching for best weighted-average on validation split (random search)...")
    ws = random_search_weights(X_train, y_train, n_iter=2000, seed=123)
    print(f"Best validation weighted-avg acc: {ws['acc']:.4f}")
    print(f"Weights: {np.round(ws['w'],4).tolist()}")

    # Evaluate best weights on held-out test split
    wp_test = X_test.dot(ws['w'])
    wp_test_preds = (wp_test >= 0.5).astype(int)
    wp_test_acc = accuracy_score(y_test, wp_test_preds)
    print(f"Weighted-average ensemble accuracy on held-out split: {wp_test_acc:.4f}")

    # also try simple weighting proportional to per-model accuracy
    accs = np.array(per_model_acc)
    prop_w = accs / accs.sum()
    prop_wp = all_probs.dot(prop_w)
    prop_preds = (prop_wp >= 0.5).astype(int)
    prop_acc = accuracy_score(labels, prop_preds)
    print(f"Proportional weights (by per-model acc) ensemble accuracy: {prop_acc:.4f}")

    # Save best weights + summary
    weights_out = {
        'model_names': model_names,
        'equal_weight_ensemble_acc': float(ensemble_acc),
        'stacking_acc': float(meta_acc),
        'best_val_weight_acc': float(ws['acc']),
        'best_val_weights': [float(x) for x in ws['w']],
        'best_test_weight_acc': float(wp_test_acc),
        'proportional_weights': [float(x) for x in prop_w],
        'proportional_weights_acc': float(prop_acc)
    }
    with open(os.path.join(WORKDIR, 'ensemble_weights_summary.json'), 'w') as f:
        json.dump(weights_out, f, indent=2)
    print(f"Saved weights summary to {os.path.join(WORKDIR, 'ensemble_weights_summary.json')}")

    # per-model vs ensemble summary
    print("Summary:")
    for name, acc in zip(model_names, per_model_acc):
        print(f"- {name}: {acc:.4f}")
    print(f"- Equal-weight ensemble: {ensemble_acc:.4f}")
    print(f"- Stacking meta-learner: {meta_acc:.4f}")

    # Save outputs
    import pandas as pd
    df = pd.DataFrame(all_probs, columns=model_names)
    df['label'] = labels
    out_csv = os.path.join(WORKDIR, 'ensemble_probs.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved per-image probabilities to {out_csv}")

if __name__ == '__main__':
    main()
