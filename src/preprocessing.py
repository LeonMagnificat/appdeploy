import tensorflow as tf
import numpy as np
import os

# Define dataset paths
TRAIN_DIR = "/Users/samenergy/Desktop/New Plant Diseases Dataset(Augmented)/Data/train"
VALID_DIR = "/Users/samenergy/Desktop/New Plant Diseases Dataset(Augmented)/Data/valid"

# Image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_dataset(directory):
    """
    Load dataset from directory and apply preprocessing steps.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
    )

    # Get class names
    class_names = dataset.class_names
    num_classes = len(class_names)

    return dataset, class_names, num_classes

def preprocess_data(dataset):
    """
    Apply image augmentation and normalization to the dataset.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
    ])

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    return dataset

def convert_to_numpy(dataset):
    """
    Convert TensorFlow dataset into NumPy arrays.
    """
    images, labels = [], []
    for img_batch, label_batch in dataset:
        images.extend(img_batch.numpy())
        labels.extend(label_batch.numpy())

    return np.array(images), np.array(labels)

# Load datasets
train_dataset, class_names, num_classes = load_dataset(TRAIN_DIR)
valid_dataset, _, _ = load_dataset(VALID_DIR)

# Apply preprocessing
train_dataset = preprocess_data(train_dataset)
valid_dataset = preprocess_data(valid_dataset)

# Convert to NumPy arrays
train_images, train_labels = convert_to_numpy(train_dataset)
valid_images, valid_labels = convert_to_numpy(valid_dataset)

# Save processed data as NumPy files
np.save("train_images.npy", train_images)
np.save("train_labels.npy", train_labels)
np.save("valid_images.npy", valid_images)
np.save("valid_labels.npy", valid_labels)

print(f"✅ Data preprocessing completed: {train_images.shape} training images, {valid_images.shape} validation images.")
