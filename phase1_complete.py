"""
PHASE 1 - CNN AUTOENCODER FOR UNSUPERVISED ANOMALY ANALYSIS
UCSD Ped1

This version fixes the main problems in the original code:
1. No Test=Anomaly assumption. Ground truth is read from *_gt folders.
2. Train/validation split is made by normal training clips.
3. The model is trained from scratch; no old model is silently loaded.
4. The decoder is strengthened to improve reconstruction quality.
5. MSE frame threshold is fixed for all test frames.
6. Pixel localization is calibrated from normal validation reconstruction errors.
7. Original and reconstructed images are never manually edited.
8. All test frames are evaluated by default.

Expected dataset structure:
UCSD_Anomaly_Dataset.v1p2/UCSDped1/
    Train/Train001/*.tif ... Train034/*.tif
    Test/Test001/*.tif ... Test036/*.tif
    Test/Test001_gt/*.bmp ... Test036_gt/*.bmp
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance"
DATASET_PATH = os.path.join(BASE_DIR, "UCSD_Anomaly_Dataset", "UCSD_Anomaly_Dataset.v1p2")
PED1_PATH = os.path.join(DATASET_PATH, "UCSDped1")

RESULTS_DIR = os.path.join(BASE_DIR, "results_phase1_fixed")
MODEL_DIR = os.path.join(BASE_DIR, "models_phase1_fixed")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 80
VALIDATION_FRACTION = 0.10
FRAME_THRESHOLD_PERCENTILE = 95.0
PIXEL_Z_THRESHOLD = 3.0
MAX_PIXEL_BASELINE_FRAMES = 1000
N_NORMAL_EXAMPLES = 15
N_ANOMALY_EXAMPLES = 15
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# FILE UTILITIES
# ============================================================
EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def list_images(folder, include_gt=False):
    folder = Path(folder)
    files = []
    if not folder.exists():
        return files
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            if not include_gt and "_gt" in p.parent.name.lower():
                continue
            files.append(str(p))
    return sorted(files, key=natural_key)


def read_frame(path):
    # UCSD frames are grayscale; replicate to 3 channels to retain 224x224x3.
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.repeat(img[..., None], 3, axis=2)


def find_gt_path(image_path):
    p = Path(image_path)
    clip_dir = p.parent
    clip_name = clip_dir.name
    if not clip_name.lower().startswith("test"):
        return None
    gt_dir = clip_dir.parent / f"{clip_name}_gt"
    if not gt_dir.exists():
        return None

    candidates = [
        gt_dir / p.name,
        gt_dir / f"{p.stem}.bmp",
        gt_dir / f"{p.stem}.png",
        gt_dir / f"{p.stem}.tif",
        gt_dir / f"{p.stem}.tiff",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    digits = re.findall(r"\d+", p.stem)
    if digits:
        number = int(digits[-1])
        for g in list_images(gt_dir, include_gt=True):
            gd = re.findall(r"\d+", Path(g).stem)
            if gd and int(gd[-1]) == number:
                return g
    return None


def read_mask(path):
    if path is None:
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    return m > 0

# ============================================================
# DATA SPLIT
# ============================================================

def split_by_clip(files, validation_fraction=0.10):
    clip_files = {}
    for f in files:
        clip = Path(f).parent.name
        clip_files.setdefault(clip, []).append(f)
    clips = sorted(clip_files, key=natural_key)
    rng = random.Random(SEED)
    rng.shuffle(clips)
    n_val = max(1, round(len(clips) * validation_fraction))
    val_clips = set(clips[:n_val])
    train, val = [], []
    for clip, fs in clip_files.items():
        (val if clip in val_clips else train).extend(fs)
    return sorted(train, key=natural_key), sorted(val, key=natural_key), sorted(val_clips, key=natural_key)

# ============================================================
# TF DATA PIPELINE
# ============================================================

def tf_read(path):
    def load(p):
        img = read_frame(p.numpy().decode("utf-8"))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.float32)
        return img.astype(np.float32)
    x = tf.py_function(load, [path], tf.float32)
    x.set_shape((224, 224, 3))
    return x


def make_dataset(files, training=False):
    ds = tf.data.Dataset.from_tensor_slices(files)
    if training:
        ds = ds.shuffle(min(len(files), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p: tf_read(p), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ============================================================
# IMPROVED CNN AUTOENCODER
# ============================================================

def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=name+"_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=name+"_conv2")(x)
    return x


def reconstruction_loss(y_true, y_pred):
    # Keep MSE dominant because MSE is the anomaly score, while a small MAE
    # component reduces excessive smoothing.
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.8 * mse + 0.2 * mae


def build_autoencoder():
    inp = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS), name="input_frame")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); latent = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(latent)
    x = layers.BatchNormalization()(x); x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.UpSampling2D(2)(x)
    out = layers.Conv2D(CHANNELS, 3, padding="same", activation="sigmoid", name="reconstructed_frame")(x)
    model = Model(inp, out, name="Phase1_CNN_Autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=reconstruction_loss, metrics=["mse"])
    return model

# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_files, val_files):
    train_ds = make_dataset(train_files, True)
    val_ds = make_dataset(val_files, False)
    best_path = os.path.join(MODEL_DIR, "best_autoencoder.keras")

    callbacks = [
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    model.save(os.path.join(MODEL_DIR, "final_autoencoder.keras"))
    return history, best_path


def save_history(history):
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("CNN Autoencoder Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# ============================================================
# RECONSTRUCTION / ERRORS
# ============================================================

def reconstruct(model, image):
    rec = model.predict(image[None, ...], verbose=0)[0]
    return np.clip(rec, 0, 1)


def mse(original, reconstructed):
    return float(np.mean(np.square(original - reconstructed)))


def pixel_error(original, reconstructed):
    return np.mean(np.square(original - reconstructed), axis=2).astype(np.float32)

# ============================================================
# THRESHOLDS
# ============================================================

def validation_errors(model, files):
    vals = []
    for f in tqdm(files, desc="Normal validation MSE"):
        img = read_frame(f)
        if img is None:
            continue
        vals.append(mse(img, reconstruct(model, img)))
    return np.asarray(vals, dtype=np.float32)


def pixel_baseline(model, val_files, max_frames=1000):
    fs = val_files
    if max_frames and len(fs) > max_frames:
        idx = np.linspace(0, len(fs)-1, max_frames, dtype=int)
        fs = [fs[i] for i in idx]

    s = np.zeros((224, 224), dtype=np.float64)
    ss = np.zeros((224, 224), dtype=np.float64)
    n = 0
    for f in tqdm(fs, desc="Normal pixel-error baseline"):
        img = read_frame(f)
        if img is None:
            continue
        e = pixel_error(img, reconstruct(model, img)).astype(np.float64)
        s += e
        ss += e * e
        n += 1
    if n < 2:
        raise RuntimeError("Not enough validation frames for pixel baseline.")
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def z_error_map(original, reconstructed, mean_map, std_map):
    raw = pixel_error(original, reconstructed)
    z = (raw - mean_map) / (std_map + 1e-6)
    z = np.maximum(z, 0)
    return cv2.GaussianBlur(z.astype(np.float32), (5, 5), 0)


def localized_heatmap(z, threshold=3.0):
    mask = (z >= threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    min_area = max(20, int(mask.size * 0.0005))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    return z * cleaned, cleaned.astype(bool)

# ============================================================
# GROUND TRUTH AND TESTING
# ============================================================

def ground_truth(image_path):
    gt_path = find_gt_path(image_path)
    if gt_path is None:
        return None, None, None
    mask = read_mask(gt_path)
    if mask is None:
        return None, gt_path, None
    return int(np.any(mask)), gt_path, mask


def test_all(model, test_files, frame_threshold):
    rows = []
    for f in tqdm(test_files, desc="Testing all Ped1 frames"):
        img = read_frame(f)
        if img is None:
            continue
        rec = reconstruct(model, img)
        score = mse(img, rec)
        pred = int(score > frame_threshold)
        gt, gt_path, _ = ground_truth(f)
        rows.append({
            "image_path": f,
            "image_name": Path(f).name,
            "clip": Path(f).parent.name,
            "mse": score,
            "threshold": frame_threshold,
            "prediction": pred,
            "prediction_label": "Anomaly" if pred else "Normal",
            "ground_truth": gt,
            "ground_truth_mask": gt_path,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "all_test_frame_results.csv"), index=False)
    return df

# ============================================================
# METRICS
# ============================================================

def metrics_report(df):
    valid = df[df.ground_truth.notna()].copy()
    if valid.empty:
        print("WARNING: no ground-truth masks matched the test frames. Metrics not calculated.")
        return None

    y_true = valid.ground_truth.astype(int).values
    y_pred = valid.prediction.astype(int).values
    scores = valid.mse.astype(float).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100),
        "precision": float(precision_score(y_true, y_pred, zero_division=0) * 100),
        "recall": float(recall_score(y_true, y_pred, zero_division=0) * 100),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0) * 100),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": cm.tolist(),
        "labeled_frames": int(len(valid)),
        "unlabeled_frames": int(len(df) - len(valid)),
    }
    print("\n===== FRAME-LEVEL RESULTS =====")
    print(f"Accuracy : {out['accuracy']:.2f}%")
    print(f"Precision: {out['precision']:.2f}%")
    print(f"Recall   : {out['recall']:.2f}%")
    print(f"F1-score : {out['f1_score']:.2f}%")
    print(f"ROC-AUC  : {out['roc_auc'] if out['roc_auc'] is not None else 'N/A'}")
    print("Confusion matrix [Normal, Anomaly]:")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))
    with open(os.path.join(RESULTS_DIR, "phase1_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

# ============================================================
# 30 FRAME VISUALIZATION
# ============================================================

def select_30(df):
    labeled = df[df.ground_truth.notna()].copy()
    normal = labeled[labeled.ground_truth == 0].sort_values("mse").head(N_NORMAL_EXAMPLES)
    anomaly = labeled[labeled.ground_truth == 1].sort_values("mse", ascending=False).head(N_ANOMALY_EXAMPLES)
    return pd.concat([normal, anomaly], ignore_index=True)


def save_visual(model, row, frame_threshold, mean_map, std_map, out_file):
    img = read_frame(row.image_path)
    if img is None:
        return
    rec = reconstruct(model, img)
    score = mse(img, rec)
    raw = pixel_error(img, rec)
    z = z_error_map(img, rec, mean_map, std_map)
    loc, _ = localized_heatmap(z, PIXEL_Z_THRESHOLD)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    ax[0].imshow(img); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(rec); ax[1].set_title(f"Reconstructed\nMSE={score:.6f}"); ax[1].axis("off")
    im1 = ax[2].imshow(raw, cmap="hot"); ax[2].set_title("Raw reconstruction error"); ax[2].axis("off")
    plt.colorbar(im1, ax=ax[2], fraction=0.046)
    im2 = ax[3].imshow(loc, cmap="hot"); ax[3].set_title(f"Localized error\nZ >= {PIXEL_Z_THRESHOLD}"); ax[3].axis("off")
    plt.colorbar(im2, ax=ax[3], fraction=0.046)

    gt = "GT: Anomaly" if row.ground_truth == 1 else "GT: Normal" if row.ground_truth == 0 else "GT: unavailable"
    pred = "Prediction: ANOMALY" if score > frame_threshold else "Prediction: NORMAL"
    fig.suptitle(f"{pred} | {gt}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()


def make_30_visuals(df, model, threshold, mean_map, std_map):
    out_dir = os.path.join(RESULTS_DIR, "30_frame_analysis")
    os.makedirs(out_dir, exist_ok=True)
    selected = select_30(df)
    selected.to_csv(os.path.join(out_dir, "selected_30_frames.csv"), index=False)
    print(f"Selected {len(selected)} labeled frames for qualitative analysis.")
    for i, (_, row) in enumerate(tqdm(selected.iterrows(), total=len(selected), desc="Saving 30-frame results"), 1):
        name = f"{i:02d}_{row.clip}_{row.image_name}.png"
        save_visual(model, row, threshold, mean_map, std_map, os.path.join(out_dir, name))

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("PHASE 1 - CNN AUTOENCODER FOR UNSUPERVISED ANOMALY ANALYSIS")
    print("Dataset: UCSD Ped1")
    print("=" * 75)

    train_root = Path(PED1_PATH) / "Train"
    test_root = Path(PED1_PATH) / "Test"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Check BASE_DIR. Expected:\n{train_root}\nand\n{test_root}"
        )

    # 1. Discover original frames directly; no JPG extraction is required.
    train_files = list_images(train_root)
    test_files = list_images(test_root)
    print(f"Training frames found: {len(train_files)}")
    print(f"Testing frames found : {len(test_files)}")

    if not train_files or not test_files:
        raise RuntimeError("No Ped1 frames were found. Check your dataset path.")

    # 2. Normal train/validation split by clip.
    train_split, val_split, val_clips = split_by_clip(train_files, VALIDATION_FRACTION)
    print(f"Training frames used : {len(train_split)}")
    print(f"Normal validation    : {len(val_split)}")
    print(f"Validation clips     : {val_clips}")

    # 3. Build and train from scratch.
    model = build_autoencoder()
    model.summary()
    history, best_path = train_model(model, train_split, val_split)
    save_history(history)

    # Always restore the best validation model.
    model = tf.keras.models.load_model(
        best_path,
        custom_objects={"reconstruction_loss": reconstruction_loss},
        compile=False
    )

    # 4. Frame-level threshold from normal validation errors.
    val_err = validation_errors(model, val_split)
    frame_threshold = float(np.percentile(val_err, FRAME_THRESHOLD_PERCENTILE))
    print(f"\nFixed frame-level threshold ({FRAME_THRESHOLD_PERCENTILE}th percentile): {frame_threshold:.8f}")
    with open(os.path.join(MODEL_DIR, "frame_threshold.pkl"), "wb") as f:
        pickle.dump(frame_threshold, f)

    # 5. Pixel-level normal baseline for localization.
    mean_map, std_map = pixel_baseline(model, val_split, MAX_PIXEL_BASELINE_FRAMES)
    np.save(os.path.join(MODEL_DIR, "pixel_error_mean.npy"), mean_map)
    np.save(os.path.join(MODEL_DIR, "pixel_error_std.npy"), std_map)

    # 6. Test all frames.
    df = test_all(model, test_files, frame_threshold)

    # 7. Quantitative metrics only when actual GT masks are found.
    metrics = metrics_report(df)

    # 8. 30-frame qualitative analysis.
    make_30_visuals(df, model, frame_threshold, mean_map, std_map)

    summary = {
        "dataset": "UCSD Ped1",
        "input_size": [224, 224, 3],
        "training_frames_found": len(train_files),
        "training_frames_used": len(train_split),
        "validation_frames": len(val_split),
        "test_frames_found": len(test_files),
        "frame_threshold_percentile": FRAME_THRESHOLD_PERCENTILE,
        "frame_threshold": frame_threshold,
        "pixel_z_threshold": PIXEL_Z_THRESHOLD,
        "metrics": metrics,
    }
    with open(os.path.join(RESULTS_DIR, "phase1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("PHASE 1 COMPLETE")
    print("=" * 75)
    print(f"Results: {RESULTS_DIR}")
    print(f"Model  : {MODEL_DIR}")
    print("Reconstructed pixels were NOT manually modified.")
    print("One fixed frame threshold is used for all test frames.")

    return model, df, metrics


if __name__ == "__main__":
    main()
