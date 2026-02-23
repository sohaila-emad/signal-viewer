📊 Signal Viewer Pro
A comprehensive multi-modal signal visualization and analysis platform supporting medical signals (ECG/EEG), acoustic signals, stock market data, and microbiome data with AI-powered abnormality detection.

https://via.placeholder.com/800x400?text=Signal+Viewer+Pro

📋 Table of Contents
Features

Project Structure

Installation

Usage

Supported File Formats

Viewers

AI Models

API Documentation

Contributing

License

✨ Features
🎯 Core Features
Multi-channel signal visualization (up to 16+ channels)

Four specialized viewers: Continuous, XOR, Polar, Recurrence

Real-time playback controls with adjustable speed

Zoom, pan, and channel management

File upload with support for multiple formats

Drag & drop interface with beautiful landing page

🧠 AI-Powered Analysis
ECG abnormality detection (5 classes: normal, arrhythmia, tachycardia, bradycardia, fibrillation)

EEG analysis with dual-model comparison:

BIOT (Transformer-based deep learning)

Random Forest (classical machine learning)

Ensemble prediction combining both models

Confidence scoring with visual indicators

📈 Supported Signal Types
Type	Description	File Formats
❤️ Medical	ECG/EEG signals	.edf, .csv, .mat
🎵 Acoustic	Audio/sound signals	.wav, .mp3
📈 Stock	Market data	.csv, .xlsx
🧬 Microbiome	Biological data	.biom, .fasta, .tsv
📁 Project Structure
text
signal-viewer/
├── frontend/                    # React application
│   ├── public/                  # Static files
│   │   └── index.html           # Main HTML template
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── Layout/          # Layout components
│   │   │   │   ├── Header.jsx   # App header
│   │   │   │   ├── LandingPage.jsx # Welcome screen
│   │   │   │   ├── FileUploader.jsx # File upload handler
│   │   │   │   └── SignalInfo.jsx # File information panel
│   │   │   ├── Viewers/         # Signal visualization components
│   │   │   │   ├── ContinuousViewer.jsx # Time-series viewer
│   │   │   │   ├── XORViewer.jsx # Pattern difference viewer
│   │   │   │   ├── PolarViewer.jsx # Circular time viewer
│   │   │   │   └── RecurrenceViewer.jsx # State space viewer
│   │   │   └── Analysis/        # AI analysis components
│   │   │       ├── AIPrediction.jsx # ECG prediction
│   │   │       └── EEGPrediction.jsx # EEG dual-model prediction
│   │   ├── services/            # API services
│   │   │   ├── api.js           # Axios configuration
│   │   │   └── signalProcessor.js # Signal processing utilities
│   │   ├── App.jsx              # Main application component
│   │   └── index.js             # Entry point
│   ├── package.json              # Frontend dependencies
│   └── .env                      # Environment variables
│
├── backend/                      # Flask application
│   ├── app/
│   │   ├── __init__.py          # Flask app factory
│   │   ├── routes/              # API routes
│   │   │   ├── medical_routes.py # Medical signal endpoints
│   │   │   ├── upload_routes.py  # File upload handling
│   │   │   └── acoustic_routes.py # Audio signal endpoints
│   │   ├── models/               # AI models
│   │   │   ├── ecg_model.py      # ECG CNN model
│   │   │   └── eeg_loader.py     # EEG BIOT/RF models
│   │   ├── services/             # Business logic
│   │   │   ├── medical_service.py # Medical signal service
│   │   │   └── eeg_service.py     # EEG analysis service
│   │   └── utils/                 # Utilities
│   │       └── signal_processing.py # Signal processing
│   ├── models/                    # Trained model files
│   │   ├── ecg_model.h5           # ECG CNN model
│   │   ├── eeg_biot_best.pt       # BIOT transformer model
│   │   └── eeg_rf_model.joblib    # Random Forest model
│   ├── data/                       # Sample data
│   │   ├── medical/                # Sample ECG/EEG
│   │   ├── acoustic/               # Sample audio
│   │   └── stock/                  # Sample stock data
│   ├── requirements.txt            # Python dependencies
│   └── run.py                      # Entry point
│
└── docker-compose.yml              # Docker configuration
🚀 Installation
Prerequisites
Node.js (v14+)

Python (3.8+)

npm or yarn

pip

Backend Setup
bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python run.py
The backend will start at http://localhost:5000

Frontend Setup
bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start React development server
npm start
The frontend will open at http://localhost:3000

Docker Setup (Optional)
bash
# Build and run with Docker Compose
docker-compose up --build
🎮 Usage
1. Upload a Signal
Select signal type (Medical, Acoustic, Stock, Microbiome)

Drag & drop file or click to browse

Wait for processing

2. Choose Viewer
Continuous: Traditional time-series view

XOR: Pattern difference detection

Polar: Circular time representation

Recurrence: State space comparison

3. Analyze with AI
For medical signals, AI automatically analyzes

View confidence scores and predictions

Compare BIOT vs Random Forest for EEG

4. Export/Share
Save visualizations as images

Export analysis reports (coming soon)

📁 Supported File Formats
Medical Signals
text
.edf  - European Data Format (EEG/ECG)
.csv  - Comma-separated values
.mat  - MATLAB files
Acoustic Signals
text
.wav  - Waveform audio
.mp3  - MPEG Audio Layer 3
Stock Data
text
.csv  - Comma-separated values
.xlsx - Excel files
Microbiome Data
text
.biom   - Biological Observation Matrix
.fasta  - FASTA sequence format
.tsv    - Tab-separated values
🎨 Viewers
1. Continuous Viewer
text
┌─────────────────────────────────┐
│ Channel 1 ━━━━━━━━━━━━━━━━━━━ │
│ Channel 2 ━━━━━━━━━━━━━━━━━━━ │
│ Channel 3 ━━━━━━━━━━━━━━━━━━━ │
└─────────────────────────────────┘
Purpose: Traditional time-series visualization

Features: Play/pause, speed control, zoom, pan

Modes: Separate (per channel) or Overlay (stacked)

2. XOR Viewer
text
Chunk 1: ━━━━━━━━━
Chunk 2: ━━━┓━━━━━
Chunk 3: ━━━━━┓━━━━
XOR:     ━━━┓━┓━━━
Purpose: Find repeating patterns and anomalies

How it works: Divides signal into chunks, overlays with XOR

Identical chunks erase, differences highlight

3. Polar Viewer
text
     ╱╲
   ╱    ╲
  ╱      ╲
 ╱        ╲
╱          ╲
Purpose: Circular time representation

Angle (θ) = Time position

Radius (r) = Signal magnitude

Modes: Cumulative (all points) or Sliding (moving window)

4. Recurrence Viewer
text
    x x   x
  x   x x
 x x   x
x   x x   x
Purpose: Find when states repeat

X-axis: Channel X at time t₁

Y-axis: Channel Y at time t₂

Point: Appears when values are similar

🧠 AI Models
ECG Model (5 Classes)
Class	Description
normal	Normal sinus rhythm
arrhythmia	Irregular heartbeat
tachycardia	Fast heart rate (>100 bpm)
bradycardia	Slow heart rate (<60 bpm)
fibrillation	Chaotic atrial activity
EEG Models (6 Classes)
Class	Description
normal	Normal brain activity
seizure	Epileptiform activity
alcoholism	Patterns from chronic alcohol
motor_abnormality	Abnormal motor cortex
mental_stress	Elevated stress levels
epileptic_interictal	Interictal discharges
Model Architecture
BIOT (Deep Learning)

Transformer-based architecture

Pre-trained on 6 datasets

Fine-tuned for specific abnormalities

Random Forest (Classical ML)

180 features extracted per window

Statistics + frequency band powers

Ensemble of 100 decision trees

📡 API Documentation
Base URL
text
http://localhost:5000/api
Endpoints
Medical Signals
Method	Endpoint	Description
GET	/medical/signals	List available signals
GET	/medical/signal/<id>	Get specific signal
POST	/medical/analyze	Analyze ECG data
POST	/medical/eeg/predict	EEG prediction with BIOT/RF
File Upload
Method	Endpoint	Description
POST	/upload	Upload signal file
Example Requests
Upload File

javascript
const formData = new FormData();
formData.append('file', file);
formData.append('type', 'medical');

fetch('/api/upload', {
  method: 'POST',
  body: formData
});
EEG Prediction

javascript
fetch('/api/medical/eeg/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    data: [[...]],  // Multi-channel data
    fs: 250,        // Sampling frequency
    temperature: 1.0 // Confidence scaling
  })
});
🛠️ Technologies Used
Frontend
React 18 - UI library

Recharts - Charting library

Axios - HTTP client

CSS3 - Styling and animations

Backend
Flask - Web framework

TensorFlow/PyTorch - Deep learning

scikit-learn - Classical ML

NumPy/SciPy - Signal processing

Pandas - Data manipulation

WFDB/PyEDFlib - Medical data formats

🤝 Contributing
Fork the repository

Create feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
BIOT model from braindecode

MIT-BIH Arrhythmia Database

PhysioNet for medical data resources

Built with ❤️ for signal processing and AI research

