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
        # fuzzy=True helps with large-file confirm pages
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
def ensemble_predict(models, img_array):
    """Return (predicted_class_index, probs_array_of_shape(n_classes,))"""
    dense121, dense169, effnet = models
    # Ensure outputs are numpy arrays, squeeze batch dim
    p169 = np.asarray(dense169.predict(img_array))
    p121 = np.asarray(dense121.predict(img_array))
    p50  = np.asarray(effnet.predict(img_array))

    # squeeze to (n_classes,)
    p169 = np.squeeze(p169)
    p121 = np.squeeze(p121)
    p50  = np.squeeze(p50)

    # In case a model returns shape (,) for single-class, ensure vector
    p169 = np.atleast_1d(p169)
    p121 = np.atleast_1d(p121)
    p50  = np.atleast_1d(p50)

    final = 0.55 * p169 + 0.30 * p121 + 0.15 * p50

    # ensure probabilities sum to ~1 (if models output logits softmax may be needed)
    # If outputs look like logits, user should have models that return softmax. We'll normalize:
    if final.sum() <= 0:
        probs = np.ones_like(final) / len(final)
    else:
        probs = final / (final.sum() + 1e-8)

    cls = int(np.argmax(probs))
    return cls, probs

# -----------------------------
# STABLE GRAD-CAM IMPLEMENTATION
# -----------------------------
def find_last_conv_layer(model):
    """Find the name of the last conv layer in a model."""
    for layer in reversed(model.layers):
        # some layers may not have output shape (e.g., FunctionalNodes), guard it
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            return layer.name
    return None

def grad_cam(model, img_array):
    """
    Compute Grad-CAM heatmap for a single input image array (batch size 1).
    Returns heatmap as 2D numpy array (H, W) corresponding to convolutional output resized later.
    """
    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        raise RuntimeError("No convolutional layer found in the model for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # predictions shape -> (1, n_classes)
        preds0 = predictions[0]
        top_class = tf.argmax(preds0)
        # use scalar loss
        loss = preds0[top_class]

    # gradients of the top predicted class w.r.t conv outputs
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # fallback: try watch variables or raise
        raise RuntimeError("Gradients returned None. Grad-CAM cannot proceed.")

    # Global average pooling on gradients (pooled gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # remove batch dim -> (H, W, C)

    # Weighted combination of forward activation maps
    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[2, 0])
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()
    # Normalize to [0,1]
    if np.max(heatmap) != 0:
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap  # shape (h, w)

def ensemble_gradcam(models, img_array):
    """Combine grad-cams from ensemble with the same weights as ensemble_predict."""
    dense121, dense169, effnet = models
    cam1 = grad_cam(dense169, img_array)
    cam2 = grad_cam(dense121, img_array)
    cam3 = grad_cam(effnet, img_array)

    # Resize cams to IMG_SIZE and combine
    cam1_r = cv2.resize(cam1, (IMG_SIZE, IMG_SIZE))
    cam2_r = cv2.resize(cam2, (IMG_SIZE, IMG_SIZE))
    cam3_r = cv2.resize(cam3, (IMG_SIZE, IMG_SIZE))

    ensemble_cam = 0.55 * cam1_r + 0.30 * cam2_r + 0.15 * cam3_r
    if ensemble_cam.max() != 0:
        ensemble_cam = ensemble_cam / (ensemble_cam.max() + 1e-8)
    else:
        ensemble_cam = np.zeros_like(ensemble_cam)
    return ensemble_cam

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

    # show sidebar with model status and instructions
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

    # Try to load models (show spinner + nice message). If fail, show error and stop.
    try:
        with st.spinner("Downloading & loading models (this runs only once)..."):
            models = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            # Read image bytes into OpenCV
            image_bytes = uploaded.read()
            img_arr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Uploaded file is not a valid image or is corrupted.")

            # Show the uploaded image small preview
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
            # ensure probs is 1D array
            probs_1d = np.asarray(probs).reshape(-1)
            # show numeric table
            prob_df = pd.DataFrame({
                "class": CLASS_NAMES,
                "probability": probs_1d
            })
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
            # Present friendly error and log details to console for debugging
            st.error(f"An error occurred while processing the image: {e}")
            # Print to server logs (visible in Streamlit Cloud logs)
            st.write("Please check logs for details.")
            print("ERROR in app:", e)

if __name__ == "__main__":
    main()
