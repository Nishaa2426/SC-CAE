import os
import random
import pickle
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)

BATCH_SIZE = 4

EPOCHS = 10

MAX_TRAIN_FRAMES = 1500
MAX_VAL_FRAMES = 300

# ============================================================
# 2. PROJECT PATH
# ============================================================

BASE_DIR = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance"

TRAIN_FOLDER = os.path.join(
    BASE_DIR,
    "output",
    "train_frames"
)

# NEW MODEL FOLDER
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_phase1_small"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_autoencoder.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_autoencoder.keras"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history.pkl"
)

CHECK_PATH = os.path.join(
    MODEL_DIR,
    "normal_reconstruction_check.png"
)

# ============================================================
# 3. GPU INFORMATION
# ============================================================

print("=" * 70)
print("PHASE 1 - CNN AUTOENCODER")
print("UCSD PED1 - SMALL DATASET VERSION")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPU devices:", gpus)

if not gpus:
    print("Running on CPU.")

# ============================================================
# 4. CHECK TRAINING FOLDER
# ============================================================

if not os.path.exists(TRAIN_FOLDER):

    raise FileNotFoundError(
        "\nTraining folder does not exist:\n"
        + TRAIN_FOLDER
    )

# ============================================================
# 5. FIND IMAGE FILES
# ============================================================

def get_image_files(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(extensions):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    files.sort()

    return files


all_files = get_image_files(
    TRAIN_FOLDER
)

print(
    "\nTotal frames found:",
    len(all_files)
)

if len(all_files) == 0:

    raise RuntimeError(
        "No image files found."
    )

# ============================================================
# 6. SHUFFLE
# ============================================================

random.Random(SEED).shuffle(
    all_files
)

# ============================================================
# 7. SELECT SMALL SUBSET
# ============================================================

required_frames = (
    MAX_TRAIN_FRAMES +
    MAX_VAL_FRAMES
)

if len(all_files) < required_frames:

    raise RuntimeError(
        f"Only {len(all_files)} frames available, "
        f"but {required_frames} are required."
    )

selected_files = all_files[
    :required_frames
]

train_files = selected_files[
    :MAX_TRAIN_FRAMES
]

val_files = selected_files[
    MAX_TRAIN_FRAMES:
]

print(
    "\nSelected training frames:",
    len(train_files)
)

print(
    "Selected validation frames:",
    len(val_files)
)

# ============================================================
# 8. PREPROCESSING
# ============================================================

def preprocess_frame(path):

    # UCSD Ped1 grayscale frame
    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        raise ValueError(
            f"Could not read: {path}"
        )

    # Resize
    img = cv2.resize(
        img,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    img = (
        img.astype(
            np.float32
        ) / 255.0
    )

    # Convert grayscale -> 3 channels
    img = np.stack(
        [
            img,
            img,
            img
        ],
        axis=-1
    )

    return img


# ============================================================
# 9. STREAMING DATA GENERATOR
# ============================================================

class AutoencoderGenerator(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        files,
        batch_size=4,
        shuffle=True
    ):

        self.files = list(files)

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.files)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.files)
                /
                self.batch_size
            )
        )

    def __getitem__(
        self,
        index
    ):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        images = []

        for i in batch_indices:

            img = preprocess_frame(
                self.files[i]
            )

            images.append(img)

        x = np.asarray(
            images,
            dtype=np.float32
        )

        # Autoencoder learns:
        # input = original frame
        # target = same original frame

        return x, x

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


train_generator = AutoencoderGenerator(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = AutoencoderGenerator(
    val_files,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# 10. CNN AUTOENCODER
# ============================================================

def build_autoencoder(
    input_shape=(224, 224, 3)
):

    input_img = layers.Input(
        shape=input_shape,
        name="input_frame"
    )

    # ========================================================
    # ENCODER
    # ========================================================

    # 224 x 224
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(input_img)

    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    # 112 x 112

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    # 56 x 56

    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    # 28 x 28

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    encoded = layers.MaxPooling2D(
        (2, 2),
        padding="same",
        name="latent_representation"
    )(x)

    # 14 x 14

    # ========================================================
    # LATENT REFINEMENT
    # ========================================================

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    )(encoded)

    # ========================================================
    # DECODER
    # ========================================================

    # 14 -> 28

    x = layers.UpSampling2D(
        (2, 2),
        interpolation="bilinear"
    )(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    # 28 -> 56

    x = layers.UpSampling2D(
        (2, 2),
        interpolation="bilinear"
    )(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    # 56 -> 112

    x = layers.UpSampling2D(
        (2, 2),
        interpolation="bilinear"
    )(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    # 112 -> 224

    x = layers.UpSampling2D(
        (2, 2),
        interpolation="bilinear"
    )(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    # ========================================================
    # RECONSTRUCTED IMAGE
    # ========================================================

    decoded = layers.Conv2D(
        3,
        (3, 3),
        activation="sigmoid",
        padding="same",
        name="reconstructed_frame"
    )(x)

    autoencoder = Model(
        input_img,
        decoded,
        name="Phase1_CNN_Autoencoder"
    )

    # MSE = reconstruction loss
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return autoencoder


# ============================================================
# 11. BUILD MODEL
# ============================================================

print("\nBuilding CNN Autoencoder...")

autoencoder = build_autoencoder()

autoencoder.summary()

# ============================================================
# 12. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )
]

# ============================================================
# 13. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING STARTED")
print("=" * 70)

history = autoencoder.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

autoencoder.save(
    FINAL_MODEL_PATH
)

# ============================================================
# 15. SAVE HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "wb"
) as f:

    pickle.dump(
        history.history,
        f
    )

# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

# ============================================================
# 17. RECONSTRUCTION TEST
# ============================================================

print(
    "\nTesting reconstruction on a normal frame..."
)

sample_path = val_files[0]

original = preprocess_frame(
    sample_path
)

reconstructed = best_model.predict(
    original[np.newaxis, ...],
    verbose=0
)[0]

# ============================================================
# 18. MSE
# ============================================================

mse = np.mean(
    np.square(
        original -
        reconstructed
    )
)

print(
    "\nNormal-frame MSE:",
    float(mse)
)

# ============================================================
# 19. SAVE ORIGINAL + RECONSTRUCTION
# ============================================================

original_gray = (
    original[:, :, 0] * 255
).astype(np.uint8)

reconstructed_gray = np.clip(
    reconstructed[:, :, 0] * 255,
    0,
    255
).astype(np.uint8)

comparison = np.hstack(
    [
        original_gray,
        reconstructed_gray
    ]
)

cv2.imwrite(
    CHECK_PATH,
    comparison
)

# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nBest model:")
print(BEST_MODEL_PATH)

print("\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nReconstruction check:")
print(CHECK_PATH)

print("\nLEFT  = ORIGINAL NORMAL FRAME")
print("RIGHT = ACTUAL RECONSTRUCTED FRAME")

print("\nIMPORTANT:")
print("The reconstructed frame is generated by the CNN decoder.")
print("It is NOT another dataset image.")