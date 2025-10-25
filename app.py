# app.py
import os
import time
import hashlib
import traceback
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---------------- Page config ----------------
st.set_page_config(page_title="Aerial Object Classification", page_icon="🕊️", layout="centered")
st.title("🕊️ Aerial Object Classification — Bird vs Drone")
st.write("Upload an aerial image to classify whether it contains a **Bird** or a **Drone**.")

# ---------------- Keras / loader ----------------
try:
    import keras  # Keras 3
    from keras.models import load_model as k3_load_model
except Exception:
    from tensorflow import keras
    from tensorflow.keras.models import load_model as k3_load_model

# POINT TO YOUR RETRAINED MODEL THAT BAKES IN: Resizing(224,224) + preprocess_input
MODEL_PATH = r"C:\Ranjith\Exercise\DS Project\aerial_project\final_model_rgb224.keras"

def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# Cache clear button
if st.button("🧼 Clear model cache"):
    st.cache_resource.clear()
    st.success("Cache cleared. Press R to rerun (or refresh the page).")

@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        stat = os.stat(MODEL_PATH)
        st.write(f"**Using model file:** `{os.path.abspath(MODEL_PATH)}`")
        st.write(f"- Size: {stat.st_size:,} bytes")
        st.write(f"- Modified: {time.ctime(stat.st_mtime)}")
        st.write(f"- SHA256: `{file_sha256(MODEL_PATH)}`")

        model = k3_load_model(MODEL_PATH, compile=False)
        st.write(f"**Loaded model.input_shape:** `{model.input_shape}`")
        return model
    except Exception:
        st.error("❌ Failed to load model. Full traceback:")
        st.code("".join(traceback.format_exc()))
        return None

model = load_trained_model()
if model is None:
    st.stop()

# --------------- Introspect model ---------------
try:
    st.write(f"**Model.input_shape:** `{model.input_shape}`")
    if hasattr(model, "layers") and model.layers:
        first = model.layers[0]
        st.write(f"**First layer:** `{first.name}` — input_shape: `{getattr(first, 'input_shape', None)}`")
except Exception as e:
    st.warning(f"Could not introspect model input shape: {e}")

# --------------- Preprocess: RGB + exactly 224×224 (NO /255, NO preprocess_input here) ---------------
def prepare_rgb_224(image_pil: Image.Image) -> np.ndarray:
    """
    Ensures RGB and exactly 224×224. Returns float32 batch (1,224,224,3).
    NOTE: Do NOT call preprocess_input here; the model already includes it.
    """
    image_pil = ImageOps.exif_transpose(image_pil).convert("RGB").resize((224, 224))
    x = np.array(image_pil, dtype=np.float32)  # keep raw 0..255; model normalizes internally
    return np.expand_dims(x, axis=0)  # (1,224,224,3)

# --------------- File uploader ---------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --------------- Quick self-test ---------------
with st.expander("🔧 Quick self-test"):
    if st.button("Run synthetic (1,224,224,3) zero-input through model"):
        try:
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            y = model.predict(dummy, verbose=0)
            st.success(f"Synthetic inference OK. Output shape: {y.shape}")
        except Exception as e:
            st.error(f"Synthetic inference failed: {e}")

# --------------- Inference pipeline ---------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        st.stop()

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prepare input (RGB + 224×224; model handles internal normalization)
    x = prepare_rgb_224(image)

    # Show quick stats
    v = x[0]
    st.write("**Input stats:**",
             f"min={float(v.min()):.4f}, max={float(v.max()):.4f}, mean={float(v.mean()):.4f}, std={float(v.std()):.4f}")
    st.write(f"**Debug — preprocessed batch shape:** `{x.shape}` (should be (1, 224, 224, 3))")

    # Accept flexible (None,None,None,3) OR exact (224,224,3)
    exp = model.input_shape
    def is_channels_last_3(shape): return isinstance(shape, (list, tuple)) and len(shape) == 4 and shape[-1] == 3
    def is_exact_224(shape): return shape[1:4] == (224, 224, 3)
    def is_flexible_rgb(shape): return (shape[1] is None) and (shape[2] is None) and (shape[3] == 3)
    ok = is_channels_last_3(exp) and (is_exact_224(exp) or is_flexible_rgb(exp))
    if not ok:
        st.error(f"❌ Shape mismatch — model expects `{exp}`.")
        st.stop()

    # ---- Class indices from training (you reported): {'bird': 0, 'drone': 1}
    # So sigmoid output is P(drone).
    th = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    # Predict
    try:
        score = float(model.predict(x, verbose=0)[0][0])  # sigmoid output for class 1 (drone)
        st.write(f"**Raw sigmoid score (P[drone]):** {score:.4f}")

        label = "Drone" if score >= th else "Bird"
        conf  = score if label == "Drone" else 1 - score

        st.markdown(f"### 🟢 Prediction: **{label}**")
        st.markdown(f"#### Confidence: **{conf * 100:.2f}%**")

        if label == "Drone":
            st.warning("⚠️ Detected object is a Drone — potential airspace alert!")
        else:
            st.success("✅ Detected object is a Bird — safe zone.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("⬆️ Upload an image to get a prediction.")
