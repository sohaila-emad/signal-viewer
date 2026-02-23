# Signal Viewer - Stock Market LSTM Prediction

## Quick Start (Local)

### First Time Setup
Run `setup.bat` to install all dependencies:
```
setup.bat
```

### Start the Application

**Option 1: Start Everything (Recommended)**
```
start_all.bat
```
This opens both backend and frontend automatically.

**Option 2: Start Separately**
```
start_backend.bat   # Backend at http://localhost:5000
start_frontend.bat  # Frontend at http://localhost:3000
```

---

## Deploy to Vercel (Frontend Only)

The frontend can be deployed to Vercel. Note: The LSTM prediction requires the backend to run.

### Steps:

1. **Install Vercel CLI** (optional):
```
bash
npm i -g vercel
```

2. **Deploy Frontend**:
```
bash
cd frontend
vercel
```

3. **Follow Prompts**:
   - Set up and deploy? → Yes
   - Which scope? → Your Vercel username
   - Want to modify settings? → No

4. **Configure API URL**:
   - Edit `frontend/src/services/api.js`
   - Change `API_BASE_URL` to your backend URL:
   
```
javascript
   const API_BASE_URL = 'https://your-backend.herokuapp.com/api';
   // or use ngrok for testing
   
```

---

## Deploy Backend to Heroku (Required for LSTM)

### Steps:

1. **Create Heroku App**:
```
bash
heroku create your-app-name
```

2. **Add Buildpacks**:
```
bash
heroku buildpacks:add heroku/python
```

3. **Deploy**:
```
bash
git push heroku main
```

4. **Set Environment**:
```
bash
heroku config:set FLASK_ENV=production
```

---

## Features

- **LSTM AI Predictions** for stock prices
- **Prediction Periods**: 7 Days, 1 Month, 6 Months
- **Confidence Intervals**: 95% with upper/lower bounds
- **Technical Indicators**: RSI, MACD, SMA, Bollinger Bands

## Usage

1. Open http://localhost:3000 in your browser
2. Go to Stock Market page (📈 tab)
3. Select a stock symbol (e.g., AAPL)
4. Click "Predict" tab
5. Select "LSTM AI" as prediction method
6. Choose prediction period (7 Days, 1 Month, 6 Months)
7. Click Predict button

## Files Created

| File | Description |
|------|-------------|
| `setup.bat` | First-time setup (create venv, install deps) |
| `start_all.bat` | Start both frontend and backend |
| `start_backend.bat` | Start backend only |
| `start_frontend.bat` | Start frontend only |

## LSTM Speed Optimizations

- Reduced epochs: 10 → 5
- Larger batch: 32 → 64
- Simplified model: 16→8 LSTM units
- Faster lookback: 20 days
