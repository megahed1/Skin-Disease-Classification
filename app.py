# app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown
import os
import pandas as pd
import altair as alt
from io import BytesIO
from PIL import Image   # ← إضافة PIL

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Skin Disease Diagnosis", layout="wide")

MODEL_IDS = {
    "dense121": "1ZbNmoMJpT9yJ3tEFxEVBAfIAhrOnq8Dx",
    "dense169": "1NjR_7DlFqM75segwvexbCFqq0qNlN0NR",
    "effnetb3": "1KU9JpiXdfW34A2L8y8Sxdz-wJbJdoQ79",
}

CLASS_NAMES = ["Eczema", "Psoriasis", "Benign Tumors", "Melanoma"]
IMG_SIZE = 224
MODEL_FILENAMES = {
    "dense121": "dense121.h5",
    "dense169": "dense169.h5",
    "effnetb3": "effnetb3.h5",
}

# -----------------------------
# UTIL: Download from Google Drive
# -----------------------------
def download_from_drive(file_id: str, out_path: str):
    if os.path.exists(out_path):
        return out_path
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, out_path, quiet=False, fuzzy=True)
        return out_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {out_path}: {e}")

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    for key, fid in MODEL_IDS.items():
        out = os.path.join("models", MODEL_FILENAMES[key])
        if not os.path.exists(out):
            download_from_drive(fid, out)

    m1 = tf.keras.models.load_model(os.path.join("models", MODEL_FILENAMES["dense121"]), compile=False)
    m2 = tf.keras.models.load_model(os.path.join("models", MODEL_FILENAMES["dense169"]), compile=False)
    m3 = tf.keras.models.load_model(os.path.join("models", MODEL_FILENAMES["effnetb3"]), compile=False)
    return m1, m2, m3

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img_rgb):
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# ENSEMBLE PREDICTION
# -----------------------------
def ensure_vector(pred):
    arr = np.asarray(pred)
    arr = np.squeeze(arr)
    arr = np.atleast_1d(arr)
    return arr

def ensemble_predict(models, img_array):
    dense121, dense169, effnet = models

    p169 = ensure_vector(dense169.predict(img_array))
    p121 = ensure_vector(dense121.predict(img_array))
    p50  = ensure_vector(effnet.predict(img_array))

    n = min(len(p169), len(p121), len(p50))
    p169 = p169[:n]; p121 = p121[:n]; p50 = p50[:n]

    final = 0.55*p169 + 0.30*p121 + 0.15*p50

    if final.sum() <= 0:
        probs = np.ones_like(final) / len(final)
    else:
        probs = final / (final.sum() + 1e-8)

    cls = int(np.argmax(probs))
    return cls, probs

# -----------------------------
# GRAD-CAM
# -----------------------------
def safe_last_conv(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            pass
    raise ValueError("No conv layer found")

def robust_gradcam(model, img_array):
    last_conv_name = safe_last_conv(model)
    last_conv = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, preds = grad_model(img_array)
        preds = tf.reshape(preds, (1, -1))
        cls = tf.argmax(preds[0])
        loss = preds[0][cls]

    grads = tape.gradient(loss, conv)

    if grads is None:
        with tf.GradientTape() as t2:
            t2.watch(conv)
            conv2, preds2 = grad_model(img_array)
            preds2 = tf.reshape(preds2, (1, -1))
            cls2 = tf.argmax(preds2[0])
            loss2 = preds2[0][cls2]
        grads = t2.gradient(loss2, conv2)
        conv = conv2

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv[0]

    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap

def ensemble_gradcam(models, img):
    dense121, dense169, effnet = models

    cam1 = robust_gradcam(dense169, img)
    cam2 = robust_gradcam(dense121, img)
    cam3 = robust_gradcam(effnet, img)

    cam1 = cv2.resize(cam1, (IMG_SIZE, IMG_SIZE))
    cam2 = cv2.resize(cam2, (IMG_SIZE, IMG_SIZE))
    cam3 = cv2.resize(cam3, (IMG_SIZE, IMG_SIZE))

    merged = 0.55*cam1 + 0.30*cam2 + 0.15*cam3

    if merged.max() > 0:
        merged /= merged.max()

    return merged

def overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.5):
    heat_uint8 = np.uint8(255 * heatmap)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    merged = cv2.addWeighted(img_rgb, 1-alpha, heat_color, alpha, 0)
    return img_rgb, merged

# -----------------------------
# UI HELPERS
# -----------------------------
def show_prob_bars(probs, class_names):
    df = pd.DataFrame({"class": class_names, "probability": probs})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("class:N", sort="-x"),
        color=alt.Color("probability:Q", scale=alt.Scale(scheme="viridis"))
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("🩺 Skin Disease Classification (Ensemble + Grad-CAM)")
    st.write("Upload a skin image...")

    with st.spinner("Loading models..."):
        models = load_models()

    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

    if uploaded:
        # --------- ⬇️ التعديل هنا فقط
        try:
            pil_img = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(pil_img)          # RGB
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            st.stop()
        # ---------

        st.subheader("Input Image")
        st.image(img_rgb, use_column_width=True)

        img_input = preprocess_image(img_rgb)

        with st.spinner("Predicting..."):
            cls_idx, probs = ensemble_predict(models, img_input)

        st.subheader(f"Prediction: **{CLASS_NAMES[cls_idx]}**")
        prob_df = pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
        st.table(prob_df)

        show_prob_bars(probs, CLASS_NAMES)

        with st.spinner("Grad-CAM..."):
            cam = ensemble_gradcam(models, img_input)

        orig, overlay = overlay_heatmap_on_image(img_rgb, cam)

        st.subheader("Grad-CAM")
        col1, col2 = st.columns(2)
        col1.image(orig, caption="Original")
        col2.image(overlay, caption="Grad-CAM Overlay")


if __name__ == "__main__":
    main()
