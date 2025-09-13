# PieceOfMind — DSU Hack 2.0

**PieceOfMind** is a multi-module research & demo project built during DSU Hack 2.0. It combines **computer vision, EEG data, and sensor-based analysis** to explore how technology can measure and improve focus, attention, and emotional awareness. The project is aimed at building smarter tools for understanding human attention and affective states using webcams, sensors, and EEG signals.

> ⚡ This README is detailed for developers, contributors, and researchers who want to set up, run, and extend the project.

---

## 🎯 Vision & Goal

Modern life is filled with distractions, and mental well-being is often neglected. The **goal of PieceOfMind** is to provide:

* 🔹 **Focus Detection**: Identify when attention drops during work/study.
* 🔹 **Emotion Recognition**: Analyze affective states using webcam-based models.
* 🔹 **EEG Integration**: Combine brainwave-based data with ML for deeper insights.
* 🔹 **Dashboard & Visualization**: Provide an easy-to-use FastAPI web app to visualize results in real-time.

This project was designed in the spirit of **hackathons**: to combine rapid prototyping, research, and practical demo pipelines.

---

## 🚀 Features

* 📷 **Camera-based Emotion & Attention Detection**

  * Pretrained PyTorch models (`best_model.pth`).
  * Webcam inference demos inside `camApp/`.

* 📊 **Sensor & EEG Analysis**

  * ML pipelines inside `SensorML/` and `eeg-ar-guide/`.
  * Includes `freq_scaler.pkl` and `raw_scaler.pkl` for preprocessing.

* 🌐 **FastAPI Web Dashboard**

  * Interactive dashboard for real-time demo.
  * Includes routes for login/auth, data visualization, and user interaction.

* 🧪 **Research-Friendly**

  * Contains notebooks (`Research/`) for prototyping.
  * Demo scripts for quick testing.

* 🔄 **Extensible Architecture**

  * Modular structure so you can plug in your own models, scalers, or datasets.

---

## 📁 Repository Structure

```
PieceOfMind-DSU-Hack-2.0/
├─ app/                  # FastAPI app code (routes, templates, static files)
├─ camApp/               # Webcam-based models, inference demos
├─ eeg-ar-guide/         # EEG + AR guides and experiments
├─ SensorML/             # ML pipeline for sensor data
├─ Research/             # Jupyter notebooks & experiments
├─ focus-catcher/        # Utilities/UI for attention focus tracking
├─ main.py               # Top-level orchestration script
├─ real_pipeline.py      # Pipeline for real-time attention/emotion inference
├─ best_model.pth        # Pretrained PyTorch model
├─ freq_scaler.pkl       # Frequency-based scaler
├─ raw_scaler.pkl        # Raw signal scaler
└─ README.md             # This file
```

---

## 🧰 Requirements

* Python **3.9+** (recommended)
* Virtual environment tool: `venv` or `conda`
* Supported OS: macOS, Linux, Windows

### Main Dependencies

```
fastapi
uvicorn
torch
torchvision
opencv-python
numpy
pandas
scikit-learn
matplotlib
jupyter
python-dotenv
pymongo
werkzeug
```

> Install from `requirements.txt` if available.

---

## ⚙️ Setup & Installation

1. **Clone the repository**

```bash
git clone https://github.com/Nithishreddyyyy/PieceOfMind-DSU-Hack-2.0.git
cd PieceOfMind-DSU-Hack-2.0
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate       # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
# Or install manually:
pip install fastapi uvicorn torch torchvision opencv-python numpy pandas scikit-learn matplotlib jupyter python-dotenv pymongo werkzeug
```

---

## ▶️ Running the Project

### 1. Start FastAPI Dashboard

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* Access in browser: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Run Camera Demo

```bash
python camApp/run_camera_demo.py
```

* Requires working webcam and `opencv-python`.
* Press `q` to quit the window.

### 3. Run Jupyter Notebooks

```bash
jupyter lab
# or
jupyter notebook
```

* Open any `.ipynb` file in `Research/` or `camApp/` to test training or experiments.

---

## 🔬 How Models Work

* **Webcam Affect Model**: Uses convolutional neural networks trained on affect/emotion datasets.
* **EEG Pipelines**: Preprocessing with scalers (`raw_scaler.pkl`, `freq_scaler.pkl`) and ML-based inference.
* **SensorML**: Processes raw sensor data, performs feature extraction, and predicts states.

---

## 🛠️ Troubleshooting

* ❌ **Torch not compiled with CUDA enabled**
  → Install CUDA-enabled PyTorch if GPU available OR run on CPU (`device = 'cpu'`).

* ❌ **Error loading ASGI app**
  → Check the path to FastAPI app (`app.main:app` vs `main:app`).

* ❌ **ngrok issues for Auth0**
  → Ngrok requires an authtoken. Either register or host via HTTPS.

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "feat: add ..."`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📜 License

This repository is provided for **educational & demo purposes**. Add a `LICENSE` file (MIT, Apache 2.0, etc.) for clarity.

---

## 📬 Contact

For collaboration or questions:

**Author:** Nithish, Manya, Arun


---

## 🏆 Acknowledgements

* Built at **DSU Hack 2.0** 🏅
* Inspired by the need to improve digital well-being & attention tracking.
* Thanks to mentors, teammates, and the open-source ML community.

---

✨ With **PieceOfMind**, we aim to build a bridge between **AI research** and **human focus** — making digital life more mindful.
