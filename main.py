import os
import glob
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. DYNAMIC DATASET LOCATION FINDER
# ------------------------------------------------------------------
csv_matches = glob.glob("/kaggle/input/**/Data_Entry_2017.csv", recursive=True)

if not csv_matches:
    raise FileNotFoundError(
        "Could not find Data_Entry_2017.csv! Please check the right sidebar in "
        "Kaggle under 'Data' and click '+ Add Data' -> Search 'NIH Chest X-ray'."
    )

CSV_PATH = csv_matches[0]
DATASET_PATH = os.path.dirname(CSV_PATH)
print(f"Found Metadata CSV at: {CSV_PATH}")

# Load CSV
df = pd.read_csv(CSV_PATH)

# List of the 14 pathology target classes
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# One-hot / Multi-binary encode labels
for disease in DISEASE_CLASSES:
    df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)

# Map image filenames across subfolders
print("Indexing image files...")
all_image_paths = {os.path.basename(x): x for x in glob.glob("/kaggle/input/**/*.png", recursive=True)}
df['file_path'] = df['Image Index'].map(all_image_paths)

# Drop missing images if any
df = df.dropna(subset=['file_path'])

# Extract numpy arrays for filenames and labels
filenames = df['file_path'].values
labels = df[DISEASE_CLASSES].values.astype(np.float32)

# Train / Validation Split (80% / 20%)
train_files, val_files, train_labels, val_labels = train_test_split(
    filenames, labels, test_size=0.2, random_state=42
)

print(f"Total Images Found: {len(df)} | Train: {len(train_files)} | Val: {len(val_files)}")

# ------------------------------------------------------------------
# 2. TF.DATA PIPELINE
# ------------------------------------------------------------------
BATCH_SIZE = 64
IMG_SIZE = (224, 224)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.io.decode_png(image_string, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image, label

def create_dataset(files, lbls, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((files, lbls))
    if is_training:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = create_dataset(train_files, train_labels, is_training=True)
val_ds = create_dataset(val_files, val_labels, is_training=False)

# ------------------------------------------------------------------
# 3. BUILD MODEL (DenseNet121)
# ------------------------------------------------------------------
base_model = tf.keras.applications.DenseNet121(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='swish', kernel_initializer='he_normal')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(14, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# ------------------------------------------------------------------
# 4. PHASE 1: TRAIN TOP HEAD
# ------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.2, patience=2, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint("best_xray_model.keras", monitor='val_auc', mode='max', save_best_only=True)
]

print("\n--- Starting Phase 1: Training Classification Head ---")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

# ------------------------------------------------------------------
# 5. PHASE 2: FINE-TUNING BACKBONE
# ------------------------------------------------------------------
base_model.trainable = True

for layer in base_model.layers[:200]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')]
)

print("\n--- Starting Phase 2: Fine-Tuning DenseNet Backbone ---")
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

print("\nTraining Complete! Best model saved as 'best_xray_model.keras'")