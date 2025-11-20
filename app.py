import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown
import os

# ============================================
# 0) Download Models from Google Drive
# ============================================

MODEL_IDS = {
    "dense121": "1ZbNmoMJpT9yJ3tEFxEVBAfIAhrOnq8Dx",
    "dense169": "1NjR_7DlFqM75segwvexbCFqq0qNlN0NR",
    "effnetb3": "1KU9JpiXdfW34A2L8y8Sxdz-wJbJdoQ79",
}

def download_models():
    os.makedirs("models", exist_ok=True)
    for name, file_id in MODEL_IDS.items():
        filename = f"models/{name}.h5"
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {name}.h5 ..."):
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                gdown.download(url, filename, quiet=False, fuzzy=True)

@st.cache_resource
def load_models():
    download_models()
    m1 = tf.keras.models.load_model("models/dense121.h5", compile=False)
    m2 = tf.keras.models.load_model("models/dense169.h5", compile=False)
    m3 = tf.keras.models.load_model("models/effnetb3.h5", compile=False)
    return m1, m2, m3

dense121, dense169, effnet = load_models()

# ============================================
# 1) Constants
# ============================================

CLASS_NAMES = ["Eczema", "Psoriasis", "Benign Tumors", "Melanoma"]
IMG_SIZE = 224

# ============================================
# 2) Preprocessing
# ============================================

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ============================================
# 3) Ensemble Prediction
# ============================================

def ensemble_predict(img):
    p169 = dense169.predict(img)
    p121 = dense121.predict(img)
    p50  = effnet.predict(img)

    final_pred = (0.55 * p169) + (0.30 * p121) + (0.15 * p50)

    cls = np.argmax(final_pred)
    return cls, final_pred

# ============================================
# 4) Grad-CAM — FIXED VERSION 🔥🔥
# ============================================

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    return None

def grad_cam(model, img_array):
    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        raise RuntimeError("No conv layer found in model.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Force predictions to shape (1, n_classes)
        predictions = tf.reshape(predictions, (1, -1))
        preds0 = predictions[0]  # vector shape (n_classes,)

        top_class = tf.argmax(preds0)
        loss = preds0[top_class]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients returned None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[2, 0])
    heatmap = tf.nn.relu(heatmap).numpy()

    heatmap /= (heatmap.max() + 1e-8)
    return heatmap


def ensemble_gradcam(img):
    cam1 = grad_cam(dense169, img)
    cam2 = grad_cam(dense121, img)
    cam3 = grad_cam(effnet, img)

    cam = (0.55 * cam1) + (0.30 * cam2) + (0.15 * cam3)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam /= (cam.max() + 1e-8)
    return cam

def overlay_gradcam(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    merged = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return img, merged

# ============================================
# 5) Streamlit UI
# ============================================

st.set_page_config(page_title="Skin Disease Diagnosis", layout="wide")

st.title("🩺 Skin Disease Classification (Ensemble + Grad-CAM)")
st.write("Upload an image to get the prediction and model explanation using Grad-CAM.")

uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        # Save uploaded file
        path = "temp.jpg"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        img_raw = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img = preprocess(img_rgb)

        # Prediction
        cls, probs = ensemble_predict(img)
        st.subheader(f"Prediction: **{CLASS_NAMES[cls]}**")

        st.write("### Probabilities")
        st.bar_chart(probs[0])

        # Grad-CAM
        with st.spinner("Generating Grad-CAM..."):
            cam = ensemble_gradcam(img)
            orig, cam_img = overlay_gradcam(path, cam)

        col1, col2 = st.columns(2)
        with col1:
            st.image(orig, caption="Original Image", use_column_width=True)
        with col2:
            st.image(cam_img, caption="Ensemble Grad-CAM", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
