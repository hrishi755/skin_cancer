import os
import json
from zipfile import ZipFile
from PIL import Image
kaggle_credentails=json.load(open("/content/kaggle (2).json"))
os.environ['KAGGLE_USERNAME']=kaggle_credentails['username']
os.environ['KAGGLE_KEY']=kaggle_credentails['key']
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
!ls
with ZipFile("/content/skin-cancer-mnist-ham10000.zip","r") as zip_ref:
  zip_ref.extractall()

# ===============================
# Skin Cancer Lesion Classification (HAM10000)
# ===============================

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===============================
# Load metadata
# ===============================
meta_path = "/content/HAM10000_metadata.csv"
skin_df = pd.read_csv(meta_path)

SIZE = 32  # Resize all images to 32x32

# Encode labels
le = LabelEncoder()
le.fit(skin_df['dx'])
skin_df['label'] = le.transform(skin_df['dx'])
print("Classes:", list(le.classes_))

# ===============================
# Balance classes
# ===============================
dfs = []
n_samples = 500  # resample size
for label in skin_df['label'].unique():
    df_class = skin_df[skin_df['label'] == label]
    df_class_balanced = resample(df_class,
                                 replace=True,
                                 n_samples=n_samples,
                                 random_state=42)
    dfs.append(df_class_balanced)

skin_df_balanced = pd.concat(dfs)
print("Balanced distribution:\n", skin_df_balanced['label'].value_counts())

# ===============================
# Auto-detect image paths
# ===============================
all_images = glob("/content/**/*.jpg", recursive=True)
print("Total images found:", len(all_images))

# Create dict: image_id -> path
image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_images}

# Map image_id to path
skin_df_balanced['path'] = skin_df_balanced['image_id'].map(image_path.get)

# Drop missing
missing = skin_df_balanced['path'].isnull().sum()
print("Missing paths:", missing)
skin_df_balanced = skin_df_balanced[skin_df_balanced['path'].notnull()]

# ===============================
# Load images safely
# ===============================
def load_image(img_path):
    try:
        return np.asarray(Image.open(img_path).resize((SIZE, SIZE)))
    except:
        return None

skin_df_balanced['image'] = skin_df_balanced['path'].map(load_image)

# Drop failed images
skin_df_balanced = skin_df_balanced[skin_df_balanced['image'].notnull()]
print("Final dataset size:", len(skin_df_balanced))

# Safety check
if len(skin_df_balanced) == 0:
    raise RuntimeError("No images matched the metadata. Check dataset path!")

# ===============================
# Train/test split
# ===============================
X = np.asarray(skin_df_balanced['image'].tolist()) / 255.0
Y = skin_df_balanced['label']
Y_cat = to_categorical(Y, num_classes=7)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y_cat, test_size=0.25, random_state=42, stratify=Y
)

print("Train samples:", x_train.shape[0], " Test samples:", x_test.shape[0])

# ===============================
# Data Augmentation
# ===============================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15
)
datagen.fit(x_train)

# ===============================
# CNN Model
# ===============================
model = Sequential([
    Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.summary()

# ===============================
# Training
# ===============================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint("best_ham10000_cnn.h5", save_best_only=True)
]

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=16),
    validation_data=(x_test, y_test),
    epochs=50,
    verbose=2,
    callbacks=callbacks
)

# ===============================
# Evaluation
# ===============================
score = model.evaluate(x_test, y_test)
print("Test accuracy:", score[1])

# Training curves
loss, val_loss = history.history['loss'], history.history['val_loss']
acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, loss, 'y', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, acc, 'y', label='Training acc')
plt.plot(epochs_range, val_acc, 'r', label='Validation acc')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Accuracy')
plt.legend()
plt.show()

# ===============================
# Confusion Matrix
# ===============================
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", linewidths=.5, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.show()


from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(image_path, threshold=0.5):
    # Load and preprocess image
    img = load_img(image_path, target_size=(SIZE, SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    # Map index back to disease label using LabelEncoder
    class_label = le.inverse_transform([class_idx])[0]

    # Handle low-confidence as "normal"
    if confidence < threshold:
        class_label = "normal"

    # Get stage + precautions
    stage, precautions = get_stage_and_precautions(class_label)

    print("=======================================")
    print(f"Prediction: {class_label} ({confidence:.2f} confidence)")
    print(f"Stage: {stage}")
    print(f"Precautions: {precautions}")
    print("=======================================")

    return class_label, stage, precautions


# ===============================
# 6. USER INPUT IMAGE PREDICTION
# ===============================
from google.colab import files

# Let user upload image
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"Predicting for: {filename}")
    class_label, stage, precautions = predict_image(filename)
    print("\nFinal Result:")
    print("Disease:", class_label)
    print("Stage:", stage)
    print("Precautions:", precautions)

# Install gradio (only once per session)
!pip install gradio

import gradio as gr

# 🔹 Replace with your existing image-processing function
# Example: if your function is `def predict(img): return "cat"`
def my_function(image):
    # image is a numpy array (H, W, C)
    # call your existing model/prediction function here
    return "Processed output"

# Create Gradio Interface for images
demo = gr.Interface(
    fn=my_function,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Textbox(label="Result")  # or gr.Image() if your function returns an image
)

# Launch the app
demo.launch(share=True)
