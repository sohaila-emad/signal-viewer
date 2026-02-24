# 🔬 Signal Viewer — Multi-Modal Biomedical & Acoustic Signal Analysis Platform

> An advanced, browser-based signal visualization and analysis platform supporting ECG/EEG medical signals, acoustic Doppler effects, drone detection, stock-market forecasting, and microbiome profiling — all powered by classical ML and pretrained deep learning models.

---

## 📸 Screenshots

### 🏠 Landing Page / Dashboard
<img width="1734" height="957" alt="25 02 2026_00 07 12_REC" src="https://github.com/user-attachments/assets/98f628c1-12a6-476e-9790-097a978bb93a" />

---

### 🫀 Medical Signal Viewer — Multi-Channel ECG

<img width="1797" height="870" alt="25 02 2026_00 15 47_REC" src="https://github.com/user-attachments/assets/467417a8-588e-4495-b4c8-da42476a9a27" />


---

### 🔁 XOR Graph Viewer
<img width="1790" height="912" alt="25 02 2026_00 16 19_REC" src="https://github.com/user-attachments/assets/bf04b350-2056-4c84-89d6-16e689988d6e" />

---

### 🧭 Polar Graph Viewer

<img width="1300" height="892" alt="25 02 2026_00 16 59_REC" src="https://github.com/user-attachments/assets/bc9d5c0e-51ac-4a6e-9136-aea4ecfb11c6" />

### 🔵 Recurrence Graph Viewer

<img width="1308" height="887" alt="25 02 2026_00 17 38_REC" src="https://github.com/user-attachments/assets/a36a8478-a175-42c6-8c74-0a23d50be01e" />


---

### 🔊 Acoustic Signals — Doppler Effect Synthesizer

<img width="1811" height="881" alt="25 02 2026_00 18 41_REC" src="https://github.com/user-attachments/assets/15ca9668-fb73-4c48-becd-20f49b6f6ed6" />


---

### 🚁 Drone Sound Detection



---

### 📈 Stock Market, Currency & Minerals Dashboard

<img width="1796" height="876" alt="25 02 2026_00 19 29_REC" src="https://github.com/user-attachments/assets/8c830462-92d0-4c8f-9511-c0c0cca4f231" />

<img width="1808" height="886" alt="25 02 2026_00 19 53_REC" src="https://github.com/user-attachments/assets/a9b6ee8e-ea6f-4b1b-b7d6-4aae54d60a9b" />
<img width="1806" height="874" alt="25 02 2026_00 20 19_REC" src="https://github.com/user-attachments/assets/8e71257c-a24e-4620-b0f0-1e1685439537" />

---

### 🧬 Microbiome Analysis Dashboard

<img width="1818" height="894" alt="25 02 2026_00 22 43_REC" src="https://github.com/user-attachments/assets/d421cc93-3405-4983-9729-6937fe9e02c8" />

<img width="1820" height="817" alt="25 02 2026_00 23 14_REC" src="https://github.com/user-attachments/assets/1a54faa5-2464-42be-a3e9-02f39a9b2bc2" />
<img width="1274" height="846" alt="25 02 2026_00 23 57_REC" src="https://github.com/user-attachments/assets/470a3ed6-817f-4646-a84f-481f4dee9578" />

<img width="1256" height="840" alt="25 02 2026_00 24 24_REC" src="https://github.com/user-attachments/assets/11974658-afc9-4e61-a080-4281532198b4" />
<img width="1288" height="841" alt="25 02 2026_00 24 58_REC" src="https://github.com/user-attachments/assets/018702a4-c3c0-4a28-80d7-193f8ad95f45" />
<img width="1325" height="857" alt="25 02 2026_00 25 16_REC" src="https://github.com/user-attachments/assets/8e4b9c0b-f278-43fb-be22-141e20e7204c" />

---

## 🗂️ Project Structure

```
signal-viewer/
│
├── frontend/                         # React application
│   ├── public/
│   │   ├── index.html                # Main HTML template
│   │   └── screenshots/              # 📸 PUT YOUR SCREENSHOTS HERE
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout/
│   │   │   │   ├── Header.jsx        # App header
│   │   │   │   ├── LandingPage.jsx   # Welcome screen
│   │   │   │   ├── FileUploader.jsx  # File upload handler
│   │   │   │   └── SignalInfo.jsx    # File information panel
│   │   │   ├── Viewers/
│   │   │   │   ├── ContinuousViewer.jsx  # Time-series viewer
│   │   │   │   ├── XORViewer.jsx         # Pattern difference viewer
│   │   │   │   ├── PolarViewer.jsx       # Circular time viewer
│   │   │   │   └── RecurrenceViewer.jsx  # State space viewer
│   │   │   ├── Pages/
│   │   │   │   ├── MedicalPage.jsx       # Medical signals page
│   │   │   │   ├── AcousticPage.jsx      # Acoustic signals page
│   │   │   │   ├── StockPage.jsx         # Stock & trading page
│   │   │   │   └── MicrobiomePage.jsx    # Microbiome analysis page
│   │   │   └── Analysis/
│   │   │       ├── AIPrediction.jsx      # ECG prediction panel
│   │   │       └── EEGPrediction.jsx     # EEG dual-model prediction
│   │   ├── services/
│   │   │   ├── api.js                    # Axios configuration
│   │   │   └── signalProcessor.js        # Signal processing utilities
│   │   ├── App.jsx                       # Main app + routing
│   │   └── index.js                      # Entry point
│   ├── package.json
│   └── .env
│
├── backend/                          # Flask application
│   ├── app/
│   │   ├── __init__.py               # Flask app factory
│   │   ├── routes/
│   │   │   ├── medical_routes.py     # ECG/EEG endpoints
│   │   │   ├── upload_routes.py      # File upload handling
│   │   │   └── acoustic_routes.py    # Audio signal endpoints
│   │   ├── models/
│   │   │   ├── ecg_model.py          # ECG CNN model
│   │   │   └── eeg_loader.py         # EEG BIOT/RF models
│   │   ├── services/
│   │   │   ├── medical_service.py    # Medical signal processing
│   │   │   └── eeg_service.py        # EEG analysis service
│   │   └── utils/
│   │       └── signal_processing.py  # Shared signal utilities
│   ├── models/                       # Trained model files
│   │   ├── ecg_model.h5              # ECG CNN (Keras)
│   │   ├── eeg_biot_best.pt          # BIOT transformer (PyTorch)
│   │   └── eeg_rf_model.joblib       # Random Forest (scikit-learn)
│   ├── data/
│   │   ├── medical/                  # Sample ECG/EEG files
│   │   ├── acoustic/                 # Sample audio files
│   │   └── stock/                    # Sample stock data
│   ├── requirements.txt
│   └── run.py
│
└── README.md
```

---

## ✨ Features Overview

### 🫀 Medical Signals (ECG / EEG)

| Feature | Description |
|---|---|
| **Signal Types** | Multi-channel ECG (12-lead) and EEG (up to 64 channels) |
| **Abnormalities** | Atrial Fibrillation, Ventricular Tachycardia, Bundle Branch Block, ST Elevation (STEMI) |
| **AI Diagnosis** | ECGNet / EfficientNet-1D multi-channel ONNX model; runs fully in-browser |
| **Classic ML** | Pan-Tompkins R-peak detector + HRV autocorrelation arrhythmia classifier |
| **Continuous Viewer** | Fixed-time viewport with play/pause, speed (0.25×–8×), zoom, and pan |
| **Multi-panel Mode** | N independent mini-viewers synchronized across speed, zoom, and pan |
| **Single-panel Mode** | All channels overlaid; per-channel show/hide, color picker, thickness slider |
| **XOR Viewer** | Chunks XOR'd pixel-by-pixel; identical signals cancel; configurable chunk width |
| **Polar Viewer** | r = amplitude, θ = time; latest-window or cumulative mode |
| **Recurrence Viewer** | CHx vs CHy phase-space cumulative scatter; multiple pairs selectable |
| **Colormap Control** | Viridis, Plasma, Jet, Magma, Inferno for 2D intensity graphs |
| **File Formats** | EDF+, CSV, MIT-BIH PhysioNet .dat/.hea |

### 🔊 Acoustic Signals

| Feature | Description |
|---|---|
| **Doppler Synthesizer** | Real-time audio generation using relativistic Doppler formula; WAV export |
| **Velocity Estimator** | Classical spectrogram peak-tracking algorithm; estimates v and f₀ from real recordings |
| **Drone Detector** | MFCC feature extraction + SVM classifier; trained on drone vs ambient audio |

### 📈 Trading Signals

| Feature | Description |
|---|---|
| **Asset Classes** | Stocks (AAPL, TSLA…), Currencies (EUR/USD, BTC/USD…), Minerals (Gold, Silver, Copper) |
| **Visualization** | Candlestick OHLC + volume bars, line chart, technical overlays (SMA, EMA, RSI, Bollinger) |
| **Forecasting** | ARIMA (short-term) + LSTM (medium-term); 7/30-day horizon |
| **Data Source** | Yahoo Finance API (live), with static fallback datasets included |

### 🧬 Microbiome Signals

| Feature | Description |
|---|---|
| **Datasets** | iHMP (Integrative Human Microbiome Project), iPOP sample data included |
| **Visualizations** | Stacked abundance bars, Shannon diversity over time, PCoA 2D scatter, correlation heatmap |
| **Disease Profiles** | IBD, Type 2 Diabetes, Colorectal Cancer, Healthy Control |
| **Patient Profiling** | Random Forest classifier on genus-level OTU features; outputs disease-state probability |

---

## 🧠 Algorithms & Models

### AI Models (In-Browser ONNX)

| Module | Model | Architecture | Input |
|---|---|---|---|
| ECG Classification | ECGNet | 1D CNN, multi-channel | 12-lead × 5000 samples |
| EEG Classification | EEGNet | Depthwise-separable CNN | N-channel × 1000 samples |
| Drone Detection | MobileNet Audio | MFCC spectrogram | 128×128 mel image |

### Classical ML Algorithms

| Task | Algorithm |
|---|---|
| Arrhythmia detection | Pan-Tompkins QRS + HRV autocorrelation |
| Doppler velocity estimation | Spectrogram ridge-tracking (STFT + quadratic peak interpolation) |
| Microbiome profiling | Random Forest on Shannon entropy + OTU relative abundances |
| Stock forecasting | ARIMA(p,d,q) auto-order selection |

---

## 🚀 Getting Started

### Prerequisites

- Modern browser (Chrome 90+, Firefox 88+, Safari 14+)
- No server required — runs fully client-side
- Optional: Node.js 18+ for local development server

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/signal-viewer.git
cd signal-viewer

# Option A: Open directly (no server needed for most features)
open index.html

# Option B: Local dev server (recommended for file upload features)
npx serve .
# → http://localhost:3000
```

### Loading Sample Data

The `samples/` directory contains ready-to-use test files:

```
samples/
├── ecg/
│   ├── normal_12lead.csv
│   ├── afib_12lead.csv
│   ├── vt_12lead.csv
│   ├── bbb_12lead.csv
│   └── stemi_12lead.csv
├── eeg/
│   ├── normal_19ch.edf
│   └── epilepsy_19ch.edf
├── acoustic/
│   ├── car_passing_real.wav
│   └── drone_flight.wav
└── microbiome/
    └── ihmp_subset.csv
```

1. Open the app in your browser
2. Navigate to the desired signal type
3. Click **Upload** or drag-and-drop a sample file
4. The viewer and AI model will initialize automatically

---

## 📐 Architecture

```
Browser
  │
  ├─ UI Layer (HTML5 + CSS3 + Vanilla JS / React components)
  │     ├─ Dashboard router
  │     ├─ Medical viewer shell
  │     ├─ Acoustic viewer shell
  │     ├─ Trading viewer shell
  │     └─ Microbiome viewer shell
  │
  ├─ Visualization Layer
  │     ├─ Canvas 2D API  (waveform drawing, XOR blending)
  │     ├─ WebGL / Three.js (polar & recurrence plots)
  │     └─ Chart.js / D3.js (stock charts, microbiome bars)
  │
  ├─ Signal Processing Layer
  │     ├─ FFT (Cooley-Tukey — pure JS)
  │     ├─ Pan-Tompkins QRS detector
  │     ├─ STFT spectrogram engine
  │     ├─ MFCC extractor
  │     └─ ARIMA solver
  │
  ├─ AI Inference Layer
  │     └─ ONNX Runtime Web (ort.js)
  │           ├─ ECGNet weights (.onnx)
  │           └─ Drone classifier weights (.onnx)
  │
  └─ Data Layer
        ├─ FileReader API (user uploads: EDF, CSV, WAV)
        ├─ Web Audio API (Doppler synthesis + playback)
        └─ Fetch API (Yahoo Finance, Alpha Vantage)
```

---

## 🧪 Signal Abnormalities — ECG

| Label | Description | Key Features |
|---|---|---|
| **Normal Sinus Rhythm** | Healthy ECG | Regular RR intervals, clear P, QRS, T waves |
| **Atrial Fibrillation (AFib)** | Disorganized atrial activity | Absent P waves, irregular RR, fibrillatory baseline |
| **Ventricular Tachycardia (VT)** | Rapid ventricular origin beats | Wide QRS (>120ms), rate 100-250 bpm, AV dissociation |
| **Left Bundle Branch Block (LBBB)** | Delayed left ventricular conduction | Wide notched QRS, negative QRS in V1, tall R in V5-V6 |
| **ST Elevation (STEMI)** | Myocardial infarction pattern | ST segment ≥1mm above baseline in ≥2 contiguous leads |

---

## 📡 Doppler Effect — Physics

The synthesizer implements the full relativistic Doppler formula:

```
Approaching: f_observed = f₀ × (v_sound + v_observer) / (v_sound - v_source)
Receding:    f_observed = f₀ × (v_sound - v_observer) / (v_sound + v_source)

where:
  f₀        = source horn frequency (Hz)
  v_sound   = 343 m/s (at 20°C)
  v_source  = vehicle velocity (m/s)
  v_observer= 0 (stationary observer)
```

The estimator recovers `f₀` and `v_source` by fitting this curve to the instantaneous frequency ridge extracted from the real recording's STFT spectrogram.

---

## 🎨 Colormap Reference

All 2D intensity graphs (XOR viewer, polar viewer heatmap, microbiome heatmap) support:

| Name | Use Case |
|---|---|
| **Viridis** | Default; perceptually uniform, colorblind-safe |
| **Plasma** | High-contrast alternative to Viridis |
| **Magma** | Dark backgrounds; good for sparse signals |
| **Inferno** | Similar to Magma, slightly warmer |
| **Jet** | Rainbow scale; legacy compatibility |

---

## 📚 Data Sources & References

### Datasets

| Module | Dataset | Source |
|---|---|---|
| ECG | PhysioNet MIT-BIH Arrhythmia DB | https://physionet.org/content/mitdb/ |
| ECG | PTB-XL 12-lead ECG DB | https://physionet.org/content/ptb-xl/ |
| EEG | CHB-MIT Scalp EEG (epilepsy) | https://physionet.org/content/chbmit/ |
| Drone audio | DCASE 2022 Task 3 | https://zenodo.org/record/6387880 |
| Stocks | Yahoo Finance API | https://finance.yahoo.com |
| Microbiome | iHMP — HMP2 | https://www.hmpdacc.org |

### Key Papers

- Pan J, Tompkins WJ. *A real-time QRS detection algorithm.* IEEE TBME, 1985.
- Hannun AY et al. *Cardiologist-level arrhythmia detection with CNNs.* Nature Medicine, 2019.
- Lawhern VJ et al. *EEGNet: A compact convolutional neural network.* J Neural Eng, 2018.
- Hintjens et al. *Doppler shift estimation from vehicle pass-by recordings.* Appl Acoustics, 2021.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/polar-viewer-improvements`
3. Commit your changes: `git commit -m 'Add cumulative polar mode with gradient fade'`
4. Push to the branch: `git push origin feature/polar-viewer-improvements`
5. Open a Pull Request

---


## 👥 Team

| Role | Responsibility |
|---|---|
| Signal Processing | ECG/EEG generation, Pan-Tompkins, Doppler math |
| AI/ML | ONNX model integration, drone classifier, stock LSTM |
| Frontend | Canvas viewers, WebGL polar plot, responsive UI |
| Data Engineering | PhysioNet/iHMP loaders, Yahoo Finance adapter |

---

*Built as part of Biomedical Signal Processing coursework — Department of Biomedical Engineering.*
