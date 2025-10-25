# 🕊️ Aerial Object Classification — Bird vs Drone

This repository contains a **fine-tuned EfficientNetB0** model to classify aerial images as **Bird** or **Drone**.

## Features
- EfficientNetB0 (ImageNet pretrained)
- RGB 224×224 input
- Internal normalization (`Resizing + preprocess_input`)
- Streamlit app for real-time inference
- Saved label map and threshold metadata

## Files
| Path | Purpose |
|------|----------|
| `models/final_model_rgb224.keras` | Main retrained model |
| `metadata/label_map.json` | Class mapping |
| `metadata/inference_meta.json` | Decision threshold |
| `app.py` | Streamlit app |
| `notebooks/DS_Project_Object_detection_MIN.ipynb` | Training notebook |

## Run App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## License
MIT © 2025 SmartRun Tech
