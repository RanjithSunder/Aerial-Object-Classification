# 🕊️ Aerial Object Classification — Bird vs Drone

This project implements a **deep learning-based aerial image classifier** that distinguishes between **Birds** and **Drones** using a **fine-tuned EfficientNetB0** model.  
It combines high accuracy with an efficient pipeline for both edge and cloud deployment.

---

## 📘 Project Summary

- **Model:** EfficientNetB0 (pretrained on ImageNet, fine-tuned on aerial dataset)
- **Input:** RGB image (224×224×3)
- **Preprocessing:** Built into the model using `Resizing(224,224)` and `preprocess_input`
- **Framework:** TensorFlow 2.19 + Keras 3.10
- **Interface:** Streamlit App for real-time inference

---

## 📂 Repository Structure

```
aerial-object-classification/
│
├── app.py                        # Streamlit inference app
├── requirements.txt               # Dependencies
├── models/
│   ├── final_model_rgb224.keras   # Final retrained model
│   ├── efficientb0_model.h5
│   └── custom_cnn_model.h5
│
├── metadata/
│   ├── label_map.json             # {'bird': 0, 'drone': 1}
│   └── inference_meta.json        # {'threshold': 0.5195}
│
└── notebooks/
    └── DS_Project_Object_detection_MIN.ipynb  # Minimal training notebook
```

---

## 🧠 Model Architecture

| Layer | Type | Output | Parameters |
|-------|------|---------|-------------|
| Input | RGB Image | (None, None, None, 3) | 0 |
| Resizing | Resizes to 224×224 | (None, 224, 224, 3) | 0 |
| EfficientNetB0 | Feature Extractor | (None, 7, 7, 1280) | 4.0M |
| GlobalAveragePooling2D | Flattened features | (None, 1280) | 0 |
| Dense | ReLU(256) | (None, 256) | 327,936 |
| Dropout | 0.4 | (None, 256) | 0 |
| Dense | Sigmoid(1) | (None, 1) | 257 |

**Total Parameters:** ≈ 4.3M  
**Trainable:** 0.3M (after freezing EfficientNet backbone)

---

## 🧩 Training Setup

- **Dataset:** `classification_dataset` with train/valid/test splits.
- **Augmentation:** rotation, shift, horizontal flip.
- **Loss:** Binary Cross-Entropy.
- **Optimizer:** Adam (`lr=1e-3`, fine-tuning at `1e-5`).
- **Callbacks:** EarlyStopping + ModelCheckpoint.

During training, the notebook automatically saves:
- `final_model_rgb224.keras`
- `label_map.json`
- `inference_meta.json`

---

## 🧪 Evaluation Metrics

| Metric | Value |
|--------|--------|
| Accuracy | 96.8% |
| AUC | 0.982 |
| Best Threshold | 0.5195 |

Confusion Matrix (sample):
```
[[208  12]
 [  9 213]]
```

---

## 🚀 Running the App

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run Streamlit app
```bash
streamlit run app.py
```

### Step 3 — Upload an image
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Output: Prediction label (Bird/Drone) and confidence percentage.

---

## 🧩 Metadata Files

| File | Purpose |
|------|----------|
| `label_map.json` | Maps labels to numeric IDs. |
| `inference_meta.json` | Stores decision threshold from validation. |

Example content:
```json
{
  "label_map": {"bird": 0, "drone": 1},
  "threshold": 0.5195
}
```

---

## 🧬 Environment Versions

| Package | Version |
|----------|----------|
| TensorFlow | 2.19.0 |
| Keras | 3.10.0 |
| NumPy | 2.0.2 |
| h5py | 3.15.1 |
| Streamlit | 1.50.0 |

---

## ⚙️ Model Reproducibility

The training notebook includes built-in reproducibility via:
```python
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
```
This ensures consistent results across runs.

---

## 📜 License

MIT License © 2025 SmartRun Tech

---

## ✨ Credits
Developed by **Immanuel L. Ebinezer**, IoT Solution Architect at **SmartRun Tech**  
For Smart Manufacturing and AI-driven Vision Solutions.

---
