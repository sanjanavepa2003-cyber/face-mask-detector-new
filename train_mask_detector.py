# ============================================================
#  FACE MASK DETECTOR — TRAINING SCRIPT
#  Run this on Google Colab (free GPU)
#  Step 1: Upload this file or paste into a Colab notebook
# ============================================================

# ── STEP 1: Install dependencies ──────────────────────────
# Run this cell first in Colab:
# !pip install tensorflow opencv-python-headless matplotlib scikit-learn

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ── STEP 2: Download dataset from Kaggle ──────────────────
# In Colab, run:
# !pip install kaggle
# Upload your kaggle.json API key, then:
# !kaggle datasets download -d andrewmvd/face-mask-detection
# !unzip face-mask-detection.zip -d dataset
#
# OR use this simpler dataset (no Kaggle account needed):
# !git clone https://github.com/prajnasb/observations
# Dataset will be at: observations/experiements/data/


# ── STEP 3: Configuration ─────────────────────────────────
INIT_LR      = 1e-4      # learning rate
EPOCHS       = 20        # number of training passes
BATCH_SIZE   = 32        # images per batch
IMG_SIZE     = 224       # MobileNetV2 expects 224x224

# Update this path to your dataset location
# Expected folder structure:
#   dataset/
#       with_mask/      ← images of people WITH masks
#       without_mask/   ← images of people WITHOUT masks
DATASET_PATH = "observations/experiements/data"


# ── STEP 4: Load and preprocess images ───────────────────
print("[INFO] Loading images...")

data   = []
labels = []

CATEGORIES = ["with_mask", "without_mask"]

for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image)
            image = preprocess_input(image)   # scales pixels to [-1, 1]
            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"  Skipping {img_name}: {e}")

data   = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"[INFO] Loaded {len(data)} images")
print(f"[INFO] with_mask:    {np.sum(labels == 'with_mask')}")
print(f"[INFO] without_mask: {np.sum(labels == 'without_mask')}")


# ── STEP 5: Encode labels & split data ───────────────────
lb = LabelBinarizer()
labels = lb.fit_transform(labels)        # with_mask=0, without_mask=1
labels = tf.keras.utils.to_categorical(labels, num_classes=2)

(trainX, testX, trainY, testY) = train_test_split(
    data, labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

# Data augmentation — creates variations of training images
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)


# ── STEP 6: Build the model ──────────────────────────────
print("[INFO] Building model...")

# Load MobileNetV2 with ImageNet weights (no top classification layer)
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

# Freeze base model — we don't retrain ImageNet weights
for layer in baseModel.layers:
    layer.trainable = False

# Add our custom classification head on top
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)                        # prevents overfitting
headModel = Dense(2, activation="softmax")(headModel)      # 2 output classes

model = Model(inputs=baseModel.input, outputs=headModel)

print(f"[INFO] Trainable layers: {sum(1 for l in model.layers if l.trainable)}")
print(f"[INFO] Frozen layers:    {sum(1 for l in model.layers if not l.trainable)}")


# ── STEP 7: Compile and train ────────────────────────────
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

print("[INFO] Training model... (this takes ~5 min on Colab GPU)")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS
)


# ── STEP 8: Evaluate ─────────────────────────────────────
print("[INFO] Evaluating model...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(
    testY.argmax(axis=1),
    predIdxs,
    target_names=lb.classes_
))


# ── STEP 9: Plot training curves ─────────────────────────
N = EPOCHS
plt.style.use("ggplot")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(np.arange(0, N), H.history["loss"],     label="train loss")
axes[0].plot(np.arange(0, N), H.history["val_loss"], label="val loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(np.arange(0, N), H.history["accuracy"],     label="train acc")
axes[1].plot(np.arange(0, N), H.history["val_accuracy"], label="val acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
print("[INFO] Saved training_plot.png")


# ── STEP 10: Save the trained model ──────────────────────
model.save("mask_detector.h5")
print("[INFO] Saved mask_detector.h5")

