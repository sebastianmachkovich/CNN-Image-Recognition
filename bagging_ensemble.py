import os
import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(input_shape=(64, 64, 3)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def gather_image_paths(train_dir):
    classes = []
    paths = []
    for cname in sorted(os.listdir(train_dir)):
        cpath = os.path.join(train_dir, cname)
        if not os.path.isdir(cpath):
            continue
        for fname in os.listdir(cpath):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.abspath(os.path.join(cpath, fname)))
                classes.append(cname)
    df = pd.DataFrame({"filename": paths, "label": classes})
    return df


def train_ensemble(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    train_df = gather_image_paths(args.train_dir)
    if train_df.empty:
        raise RuntimeError(f"No images found in {args.train_dir}")

    datagen_train = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True, zoom_range=0.1, rotation_range=10)
    datagen_val = ImageDataGenerator(rescale=1.0 / 255.0)

    val_gen = datagen_val.flow_from_directory(
        args.val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="binary",
        shuffle=False,
    )

    saved_models = []
    for i in range(args.n_estimators):
        print(f"\nTraining estimator {i + 1}/{args.n_estimators}...")
        boot_df = train_df.sample(frac=1.0, replace=True, random_state=args.seed + i).reset_index(drop=True)

        train_gen = datagen_train.flow_from_dataframe(
            boot_df,
            x_col="filename",
            y_col="label",
            target_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            class_mode="binary",
            shuffle=True,
        )

        model = build_model(input_shape=(args.img_size, args.img_size, 3))

        steps_per_epoch = math.ceil(len(boot_df) / args.batch_size)
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            verbose=1,
        )

        model_path = os.path.join(args.output_dir, f"model_{i+1}.h5")
        model.save(model_path)
        saved_models.append(model_path)
        print(f"Saved estimator to {model_path}")

    # Ensemble evaluation (probability averaging)
    print("\nEvaluating ensemble on validation set...")
    # reload models to free memory
    preds = []
    for mp in saved_models:
        m = keras.models.load_model(mp)
        val_gen.reset()
        p = m.predict(val_gen, verbose=0).ravel()
        preds.append(p)

    avg_pred = np.mean(preds, axis=0)
    y_pred = (avg_pred >= 0.5).astype(int)
    y_true = val_gen.classes

    acc = np.mean(y_pred == y_true)
    print(f"Ensemble accuracy on validation set: {acc:.4f}")

    # Optionally save averaged predictions
    out_csv = os.path.join(args.output_dir, "ensemble_preds.csv")
    files = [os.path.basename(path) for path in val_gen.filenames]
    df_out = pd.DataFrame({"file": files, "y_true": y_true, "y_prob": avg_pred, "y_pred": y_pred})
    df_out.to_csv(out_csv, index=False)
    print(f"Saved ensemble predictions to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="training_set", help="Path to training folder with class subfolders")
    p.add_argument("--val_dir",   default="test_set",     help="Path to validation/test folder with class subfolders")
    p.add_argument("--n_estimators", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--output_dir", default="ensemble_output")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_ensemble(args)
