# Signal Viewer — Local setup

Short guide to run the backend and frontend locally.

## Prerequisites

- Python 3.8+ with `venv` and `pip`
- Node.js 16+ (includes `npm`) or Yarn
- Git (optional)

## Backend (Python)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

3. Run the backend (uses Socket.IO):

```powershell
# optionally set PORT (default 5000)
$env:PORT = 5000
python backend\run.py
```

The backend listens on port `5000` by default (http://localhost:5000).

## Frontend (React)

1. Install dependencies and start dev server:

```powershell
cd frontend
npm install
npm start
```

The development server runs on port `3000` by default (http://localhost:3000).

If the frontend is configured to proxy API requests to the backend, the `proxy` field in `frontend/package.json` should point to the backend URL (for example `http://localhost:5000`).

## Accessing the app

- Open the frontend dev URL: http://localhost:3000
- The frontend will call the backend API (default http://localhost:5000) — ensure both servers are running.

## Common notes / troubleshooting

- If ports are in use, change the backend `PORT` environment variable or the frontend dev port (see `package.json` for `start` options).
- On Windows PowerShell, use `Activate.ps1` to activate the venv; on Cmd use `venv\Scripts\activate.bat`.
- Production: build the frontend with `npm run build` and serve `frontend/build` from a static server or integrate with the backend as appropriate.

If you want, I can also add a single `start` script that runs both backend and frontend concurrently or create a short `run.sh`/`run.ps1` wrapper.
