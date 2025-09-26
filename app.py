# app.py
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

import streamlit as st

# ===============================
# CONFIG
# ===============================
SIZE = 32
MODEL_PATH = "best_ham10000_cnn.h5"

# ===============================
# LOAD + PREPARE DATA
# ===============================
@st.cache_data
def load_metadata(uploaded_file=None):
    meta_path = "HAM10000_metadata.csv"

    if uploaded_file is not None:
        skin_df = pd.read_csv(uploaded_file)
    elif os.path.exists(meta_path):
        skin_df = pd.read_csv(meta_path)
    else:
        return None, None

    # Encode labels
    le = LabelEncoder()
    skin_df['label'] = le.fit_transform(skin_df['dx'])

    # Balance classes
    dfs = []
    for label in skin_df['label'].unique():
        df_class = skin_df[skin_df['label'] == label]
        df_bal = resample(df_class, replace=True, n_samples=500, random_state=42)
        dfs.append(df_bal)

    skin_df_balanced = pd.concat(dfs)

    # Map image paths (optional, if dataset images exist)
    all_images = glob("**/*.jpg", recursive=True)
    image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_images}
    skin_df_balanced['path'] = skin_df_balanced['image_id'].map(image_path.get)
    skin_df_balanced = skin_df_balanced[skin_df_balanced['path'].notnull()]

    return skin_df_balanced, le

# ===============================
# BUILD / LOAD MODEL
# ===============================
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

def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return build_model()

# ===============================
# PREDICTION FUNCTION
# ===============================
def get_stage_and_precautions(label):
    mapping = {
        "nv": ("Stage 1", "Regular skin check"),
        "mel": ("Stage 2", "Consult dermatologist ASAP"),
        "bkl": ("Stage 1", "Use sunscreen"),
        "bcc": ("Stage 2", "See doctor"),
        "akiec": ("Stage 2", "Early treatment needed"),
        "df": ("Stage 1", "Low risk"),
        "vasc": ("Stage 1", "Monitor only")
    }
    return mapping.get(label, ("Unknown", "No precautions available"))

def predict_image(model, image, le, threshold=0.5):
    img = image.resize((SIZE, SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    class_label = le.inverse_transform([class_idx])[0]

    if confidence < threshold:
        class_label = "normal"

    stage, precautions = get_stage_and_precautions(class_label)
    return class_label, confidence, stage, precautions

# ===============================
# STREAMLIT UI
# ===============================
st.title("🩺 Skin Cancer Classification (HAM10000)")

# Upload metadata file
uploaded_csv = st.file_uploader("Upload HAM10000_metadata.csv", type=["csv"])
skin_df_balanced, le = load_metadata(uploaded_csv)

if skin_df_balanced is None:
    st.error("⚠️ Metadata file not found. Please upload HAM10000_metadata.csv.")
else:
    model = get_model()

    st.markdown("Upload a skin lesion image and get predictions.")

    uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, conf, stage, precautions = predict_image(model, image, le)

        st.success(f"Prediction: **{label}** ({conf:.2f} confidence)")
        st.info(f"Stage: {stage}")
        st.warning(f"Precautions: {precautions}")
