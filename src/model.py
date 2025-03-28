import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load preprocessed data safely
def load_data(file_path):
    try:
        return np.load(file_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found -> {file_path}")
        exit(1)

train_images = load_data("/Users/samenergy/Documents/Projects/PlantDiseaseDetection/src/preprocessed data/train_images.npy")
train_labels = load_data("/Users/samenergy/Documents/Projects/PlantDiseaseDetection/src/preprocessed data/train_labels.npy")
valid_images = load_data("/Users/samenergy/Documents/Projects/PlantDiseaseDetection/src/preprocessed data/valid_images.npy")
valid_labels = load_data("/Users/samenergy/Documents/Projects/PlantDiseaseDetection/src/preprocessed data/valid_labels.npy")

# Get input shape and number of classes
input_shape = train_images.shape[1:]  # Expected (height, width, channels) -> (128, 128, 3)
num_classes = train_labels.shape[1]   # 38 classes

# Model architecture
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),  # ✅ Fixed input shape issue

        Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        Conv2D(32, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=2, strides=2),

        Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        Conv2D(64, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=2, strides=2),

        Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        Conv2D(128, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=2, strides=2),

        Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        Conv2D(256, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=2, strides=2),

        Conv2D(512, kernel_size=3, padding='same', activation='relu'),
        Conv2D(512, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=2, strides=2),

        Dropout(0.25),  # Reduce overfitting

        Flatten(),
        Dense(1500, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # ✅ Fixed optimizer issue
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create and summarize the model
model = create_cnn_model(input_shape, num_classes)
model.summary()

# Train the model
EPOCHS = 10
history = model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=EPOCHS)

# Save trained model
model.save("plant_disease_model.keras")
print("✅ Model saved as 'plant_disease_model.keras'")

# Model evaluation
train_loss, train_acc = model.evaluate(train_images, train_labels)
valid_loss, valid_acc = model.evaluate(valid_images, valid_labels)

print(f"🔍 Training Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}")

# Predictions
predictions = model.predict(valid_images)
predicted_categories = np.argmax(predictions, axis=1)
true_categories = np.argmax(valid_labels, axis=1)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(true_categories, predicted_categories))

# Confusion Matrix
cm = confusion_matrix(true_categories, predicted_categories)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Plant Disease Prediction")
plt.show()
