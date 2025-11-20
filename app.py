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
from PIL import Image

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
# UTIL: Download from Google Drive (if needed)
# -----------------------------
def download_from_drive(file_id: str, out_path: str):
    """Download a file from Google Drive using gdown if it doesn't exist."""
    if os.path.exists(out_path):
        return out_path
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, out_path, quiet=False, fuzzy=True)
        return out_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {out_path} from Drive: {e}")

# -----------------------------
# MODEL LOADING (cached)
# -----------------------------
@st.cache_resource
def load_models():
    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)
    # Download if missing
    for key, fid in MODEL_IDS.items():
        out = os.path.join("models", MODEL_FILENAMES[key])
        if not os.path.exists(out):
            download_from_drive(fid, out)

    # Load models (compile=False for faster loading)
    try:
        m1 = tf.keras.models.load_model(os.path.join("models", MODEL_FILENAMES["dense121"]), compile=False)
        m2 = tf.keras.models.load_model(os.path.join("models", MODEL_FILENAMES["dense169"]), compile=False)
        m3 = tf.keras.models.load_model(os.path.join("models", MODEL_FILENAMES["effnetb3"]), compile=False)
    except Exception as e:
        # Re-raise as runtime error for UI to catch
        raise RuntimeError(f"Error while loading models: {e}")

    return m1, m2, m3

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img_bgr):
    """Resize + normalize, return shape (1, IMG_SIZE, IMG_SIZE, 3)"""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    # convert to RGB expected by models
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# ENSEMBLE PREDICTION
# -----------------------------
def ensure_vector(pred):
    """Ensure prediction is 1D vector (n_classes,) even if model returns (n,) or scalar."""
    arr = np.asarray(pred)
    arr = np.squeeze(arr)  # remove batch dim if present
    arr = np.atleast_1d(arr)  # ensure vector
    return arr

def ensemble_predict(models, img_array):
    """Return (predicted_class_index, probs_array_of_shape(n_classes,))"""
    dense121, dense169, effnet = models

    # get raw predictions
    p169 = ensure_vector(dense169.predict(img_array))
    p121 = ensure_vector(dense121.predict(img_array))
    p50 = ensure_vector(effnet.predict(img_array))

    # If lengths mismatch, try to align by taking min length (defensive)
    n = min(len(p169), len(p121), len(p50))
    p169 = p169[:n]; p121 = p121[:n]; p50 = p50[:n]

    final = 0.55 * p169 + 0.30 * p121 + 0.15 * p50

    # normalize to probabilities
    if final.sum() <= 0:
        probs = np.ones_like(final) / len(final)
    else:
        probs = final / (final.sum() + 1e-8)

    cls = int(np.argmax(probs))
    return cls, probs

# -----------------------------
# ROBUST GRAD-CAM IMPLEMENTATION
# -----------------------------
def safe_last_conv(model):
    """Find the last 4D layer (Conv or feature map)."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("No 4D conv layer found in model.")

def robust_gradcam(model, img_array):
    """
    Robust Grad-CAM:
    - finds last conv layer safely
    - ensures predictions shape handled
    - retries gradient watching if grads are None
    Returns heatmap (2D numpy) normalized to [0,1]
    """
    last_conv_name = safe_last_conv(model)
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    # First attempt
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        # normalize preds shape to (1, n)
        preds = tf.reshape(preds, (1, -1))
        preds0 = preds[0]
        top_class = tf.argmax(preds0)
        loss = preds0[top_class]

    grads = tape.gradient(loss, conv_outputs)

    # If gradients None, try watching conv_outputs directly
    if grads is None:
        with tf.GradientTape() as tape2:
            tape2.watch(conv_outputs)
            conv_outputs2, preds2 = grad_model(img_array)
            preds2 = tf.reshape(preds2, (1, -1))
            preds0_2 = preds2[0]
            top_class2 = tf.argmax(preds0_2)
            loss2 = preds0_2[top_class2]
        grads = tape2.gradient(loss2, conv_outputs2)
        conv_outputs = conv_outputs2  # use the version from second tape

    if grads is None:
        raise RuntimeError("Gradients are None; cannot compute Grad-CAM for this model.")

    # pooled grads and weighted sum
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)

    heatmap = tf.nn.relu(heatmap).numpy()
    if np.max(heatmap) != 0:
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap

def ensemble_gradcam(models, img_array):
    """Combine grad-cams from ensemble with the same weights as ensemble_predict."""
    dense121, dense169, effnet = models

    cam169 = robust_gradcam(dense169, img_array)
    cam121 = robust_gradcam(dense121, img_array)
    cam3 = robust_gradcam(effnet, img_array)

    cam169_r = cv2.resize(cam169, (IMG_SIZE, IMG_SIZE))
    cam121_r = cv2.resize(cam121, (IMG_SIZE, IMG_SIZE))
    cam3_r = cv2.resize(cam3, (IMG_SIZE, IMG_SIZE))

    merged = 0.55 * cam169_r + 0.30 * cam121_r + 0.15 * cam3_r
    if merged.max() != 0:
        merged = merged / (merged.max() + 1e-8)
    else:
        merged = np.zeros_like(merged)
    return merged

def overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.5):
    """Overlay heatmap (0..1) onto BGR image and return RGB merged image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    merged = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return img_rgb, merged

# -----------------------------
# UI HELPERS
# -----------------------------
def show_prob_bars(probs, class_names):
    """Display probability bars using Altair for prettier colors."""
    df = pd.DataFrame({
        "class": class_names,
        "probability": probs
    })
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('probability:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('class:N', sort='-x'),
        color=alt.Color('probability:Q', scale=alt.Scale(scheme='viridis'))
    ).properties(height=150)
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# MAIN UI
# -----------------------------
def main():
    st.title("🩺 Skin Disease Classification (Ensemble + Grad-CAM)")
    st.write("Upload a skin image (jpg/png). The app will download models (if missing), run ensemble prediction and show Grad-CAM explanation.")

    # Sidebar
    with st.sidebar:
        st.header("Instructions")
        st.write("""
        1. Upload a clear image of the skin lesion.
        2. Wait for model download (first time).
        3. View prediction and Grad-CAM.
        """)
        st.markdown("---")
        st.write("Class names (order matters):")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{i} — {name}")

    # Load models
    try:
        with st.spinner("Downloading & loading models (this runs only once)..."):
            models = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            # Read image using PIL to avoid OpenCV decode issues
            try:
                pil_img = Image.open(uploaded).convert("RGB")
            except Exception as e:
                st.error(f"Error reading image: {e}")
                st.stop()
                return

            # Convert to NumPy RGB then to BGR to keep the rest of the code unchanged
            img_rgb_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)

            # Show the uploaded image preview
            st.subheader("Input Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Preprocess
            img_input = preprocess_image(img_bgr)

            # Prediction with spinner
            with st.spinner("Running ensemble prediction..."):
                cls_idx, probs = ensemble_predict(models, img_input)

            # Show results
            st.subheader(f"Prediction: **{CLASS_NAMES[cls_idx]}**")
            st.write("Probabilities:")
            probs_1d = np.asarray(probs).reshape(-1)
            prob_df = pd.DataFrame({"class": CLASS_NAMES, "probability": probs_1d})
            st.table(prob_df.style.format({"probability": "{:.4f}"}))
            # show colored bars
            show_prob_bars(probs_1d, CLASS_NAMES)

            # Grad-CAM with spinner
            with st.spinner("Computing Grad-CAM explanation..."):
                cam = ensemble_gradcam(models, img_input)

            orig_rgb, cam_overlay = overlay_heatmap_on_image(img_bgr, cam, alpha=0.5)
            st.markdown("**Grad-CAM Explanation**")
            col1, col2 = st.columns(2)
            col1.image(orig_rgb, caption="Original", use_column_width=True)
            col2.image(cam_overlay, caption="Grad-CAM Overlay", use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
            st.write("Please check logs for details.")
            print("ERROR in app:", e)

if __name__ == "__main__":
    main()
