# app.py
import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import streamlit as st

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical

# ===============================
# CONFIG
# ===============================
SIZE = 32
MODEL_PATH = "best_ham10000_cnn.h5"
ENCODER_PATH = "label_encoder.pkl"
N_SAMPLES_PER_CLASS = 50  # Reduce for faster training in Streamlit

# ===============================
# HELPER FUNCTIONS
# ===============================
def load_metadata():
    meta_path = "HAM10000_metadata.csv"
    if not os.path.exists(meta_path):
        st.error("❌ HAM10000_metadata.csv not found. Please upload it.")
        st.stop()

    skin_df = pd.read_csv(meta_path)
    le = LabelEncoder()
    skin_df['label'] = le.fit_transform(skin_df['dx'])

    # Balance classes (resample)
    dfs = []
    for label in skin_df['label'].unique():
        df_class = skin_df[skin_df['label'] == label]
        df_bal = resample(df_class, replace=True, n_samples=N_SAMPLES_PER_CLASS, random_state=42)
        dfs.append(df_bal)
    skin_df_balanced = pd.concat(dfs)

    # Map image paths
    all_images = glob("**/*.jpg", recursive=True)
    image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_images}
    skin_df_balanced['path'] = skin_df_balanced['image_id'].map(image_path.get)
    skin_df_balanced = skin_df_balanced[skin_df_balanced['path'].notnull()]

    if len(skin_df_balanced) == 0:
        st.error("❌ No images found. Please upload dataset images in the repo.")
        st.stop()

    return skin_df_balanced, le

def build_model():
    model = Sequential([
        Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)),
        MaxPool2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(32, activation="relu"),
        Dense(7, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model

def train_model(skin_df_balanced, le):
    X, y = [], []
    for path, label in zip(skin_df_balanced['path'], skin_df_balanced['label']):
        if not os.path.exists(path):
            continue
        img = load_img(path, target_size=(SIZE, SIZE))
        img_array = img_to_array(img) / 255.0
        X.append(img_array)
        y.append(label)

    if len(X) == 0:
        st.error("❌ No valid images found for training.")
        st.stop()

    X = np.array(X)
    y = to_categorical(y, num_classes=len(le.classes_))

    model = build_model()
    st.info("⚡ Training model… This may take a few minutes.")
    model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2, verbose=2)

    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    st.success("✅ Model trained and saved!")

    return model

def get_stage_and_precautions(label):
    mapping = {
        "nv": ("Stage 1", "Regular skin check"),
        "mel": ("Stage 2", "Consult dermatologist ASAP"),
        "bkl": ("Stage 1", "Use sunscreen"),
        "bcc": ("Stage 2", "See doctor"),
        "akiec": ("Stage 2", "Early treatment needed"),
        "df": ("Stage 1", "Low risk"),
        "vasc": ("Stage 1", "Monitor only"),
        "normal": ("Stage 0", "No action required"),
    }
    return mapping.get(label, ("Unknown", "No precautions available"))

def predict_image(model, image, le, threshold=0.5):
    img = image.resize((SIZE, SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = float(preds[0][class_idx])

    class_label = le.inverse_transform([class_idx])[0]
    if confidence < threshold:
        class_label = "normal"

    stage, precautions = get_stage_and_precautions(class_label)
    return class_label, confidence, stage, precautions

# ===============================
# STREAMLIT UI
# ===============================
st.title("🩺 Skin Cancer Classification (HAM10000)")

skin_df_balanced, le = load_metadata()

# Load or train model
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    st.success("✅ Loaded pre-trained model.")
else:
    model = train_model(skin_df_balanced, le)

# Upload image for prediction
uploaded_file = st.file_uploader("📤 Upload Skin Lesion Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, conf, stage, precautions = predict_image(model, image, le)

    st.success(f"✅ Prediction: **{label}** ({conf:.2f} confidence)")
    st.info(f"🩺 Stage: {stage}")
    st.warning(f"⚠️ Precautions: {precautions}")
