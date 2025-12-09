# PriceWise – Computer Price Assistant

PriceWise is a full-stack application that helps users explore the laptop market, predict fair prices for custom configurations, discover similar offers, and receive conversational advice. The project combines a FastAPI backend (CatBoost + SHAP explainability + prescriptive search) with a modern React/Tailwind UI built in Vite.

> **Live user journey:** Overview dashboard ➜ Configurable price predictor ➜ SHAP insights ➜ Similar offers finder (prefilled from last prediction) ➜ Optional chat advisor powered by GPT specs extraction.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Backend Setup](#backend-setup)
    - [Frontend Setup](#frontend-setup)
    - [Environment Variables](#environment-variables)
    - [Running Locally](#running-locally)
    - [Running with Docker](#running-with-docker)
5. [Usage Guide](#usage-guide)
6. [API Reference (Quick Look)](#api-reference-quick-look)
7. [Testing & Quality](#testing--quality)
8. [Deployment Notes](#deployment-notes)
9. [Troubleshooting](#troubleshooting)

---

## Key Features

- **Interactive Market Overview:** Price distributions, brand mix, RAM breakdown, and clustering insights derived from ~6k laptop listings.
- **Price Predictor:** Simple and advanced modes that feed a CatBoost regressor; SHAP explanations highlight drivers per prediction.
- **Similar Offers Finder:** Prescriptive search that reuses prediction inputs to surface k-nearest listings with price deltas and similarity scores.
- **Chat Advisor:** Optional GPT-powered assistant that infers specs from natural language and routes them through the same prediction stack.
- **Feedback Capture:** Inline widget records sentiment, sliders, and context for dashboard, predictor, similar, and chat experiences.
- **Unified Backend:** FastAPI serves all predictive/descriptive/prescriptive endpoints consumed by the React UI and any external clients.

---

## Architecture

```
React (Vite + Tailwind)  --->  FastAPI  --->  CatBoost model + SHAP
    |                           |                    |
    |----> Predictive UI -------|----> analytics --->|----> Feature explanations
    |----> Similar search ------|----> prescriptive metrics & distance calculations
    |----> Chat advisor ------- GPT for spec extraction / use-case inference
```

- **Frontend:** `/frontend` (React 18, Vite, Tailwind, Framer Motion, Recharts).
- **Backend:** `/backend` + `/src` modules (FastAPI endpoints, analytics utilities, CatBoost model wrappers, LLM advisor).
- **Data/Models:** stored under `/data` and `/models` (not shipped publicly).

---

## Repository Layout

| Path | Description |
| --- | --- |
| `backend/` | FastAPI application entry point (`backend/main.py`) plus ASGI config. |
| `src/` | Shared Python modules: model inference, analytics, feature engineering, chat advisor. |
| `frontend/` | React/Vite client. Entry point `src/main.tsx`, routes in `src/App.jsx`. |
| `models/` | Serialized CatBoost model artifacts (CatBoost, SHAP values, benchmark caches). |
| `data/` | Processed datasets and benchmark lookups used by analytics endpoints. |
| `docker-compose.yml`, `Dockerfile`, `Dockerfile.backend` | Containerization assets. |
| `requirements.txt` | Python dependencies (backend + analytics). |
| `test_features.py` | Example pytest module for feature pipeline. |
| `docs/*.md` | Historical notes on feature engineering, integration, and optimization. |

---

## Getting Started

### Requirements
- Python **3.11**
- Node.js **18+** (Vite requires native ESM support)
- npm **9+**
- Optional: Docker / Docker Compose

### Backend Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Environment Variables

| Variable | Description | Where |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required by chat advisor to call GPT. | Backend env (`.env`, shell, or deployment secret). |
| `FRONTEND_URL` | (Optional) URL allowed by backend CORS. | Backend env. |
| `PORT` | Backend listening port (defaults to `8000`). | Backend env. |
| `VITE_API_TARGET` | Base URL of FastAPI (defaults to `http://localhost:8000`). | Frontend `.env` or `.env.docker`. |

Create a `.env` file at repo root if needed:
```
OPENAI_API_KEY=sk-...
FRONTEND_URL=http://localhost:5173
```

### Running Locally

1. **Backend**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   FastAPI docs: http://localhost:8000/docs

2. **Frontend**
   ```bash
   cd frontend
   npm run dev
   ```
   Access UI: http://localhost:5173

### Running with Docker

```bash
docker-compose up --build
```

- Frontend served from `localhost:5173`.
- Backend reachable at `localhost:8000`.
- Compose automatically mounts source directories for live reload.

---

## Usage Guide

1. **Overview tab:** Inspect descriptive analytics (market stats, brand mix, cluster cards). Use this to decide on specs/budget.
2. **Predict tab:** Pick a use case (Simple mode) or dive into advanced sliders (screen size, refresh rate, GPU memory, etc.). Click **Predict Price**. You will see:
   - Estimated range and confidence badge.
   - SHAP breakdown for that prediction.
   - Global feature importance chart.
3. **Similar tab:** Opens a form pre-filled with the last prediction. Adjust filters if necessary and click **Find Similar Offers** to get k-best matches with “vs. predicted” deltas and similarity scores.
4. **Chat tab:** Provide natural language requirements. GPT extracts specs, sends them through the predictor, and replies with reasoning or additional tips.
5. **Feedback widget:** Appears beneath predictor and similar offers; use it to capture qualitative feedback for analytics.

---

## API Reference (Quick Look)

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Service health probe. |
| `POST /api/predict` | Direct price prediction (base request model). |
| `GET /api/predictive/options` | Allowed categorical options (brands, CPU families, etc.). |
| `POST /api/predictive/predict` | Extended prediction using the same model with more features. |
| `GET /api/statistics` & `/api/descriptive/*` | Aggregate stats for dashboard. |
| `GET /api/clusters` | Cluster metadata + scatter data. |
| `POST /api/prescriptive/similar` or `/api/similar` | k-most similar laptops to a target spec. |
| `POST /api/chat` | Chat advisor; requires `api_key` in payload. |
| `POST /api/feedback` | Store user feedback (feature + rating + context). |
| `GET /api/admin/feedback*` | Administrative endpoints to inspect feedback/workload. |

Standard FastAPI docs (`/docs`) provide request/response schemas.

---

## Testing & Quality

- **Backend tests:** `pytest test_features.py`
- **Type checking / linting:** use your preferred toolchain (e.g., `ruff`, `mypy`) if installed.
- **Frontend:** Vite includes ESLint if configured; run `npm run lint` (optional).

Continuous testing is recommended before pushing to avoid regressions; inference latency is <1 ms per sample and model weight is ~5 MB, so unit tests run quickly.

---

## Deployment Notes

- **Dockerfile / Dockerfile.backend:** Multi-stage builds (frontend bundle + Python runtime). Works with Fly.io, Railway, Render, etc.
- **Procfile:** `web: python ... uvicorn backend.main:app` for Heroku-style deployments.
- **nixpacks.toml:** optimized build hint for Railway/Nixpacks.
- **railway.json:** example configuration referencing the same FastAPI entry point.

When deploying, remember to:
- Provide `OPENAI_API_KEY` and other secrets in the hosting platform.
- Set `VITE_API_TARGET` during frontend build (Dockerfile already passes `VITE_API_TARGET=http://backend:8000` for Compose).
- Serve the compiled frontend (Vite build) through the backend or a static host if desired; `backend/main.py` exposes `/` when the `frontend/dist` directory exists.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `CORS` errors in browser | Ensure `FRONTEND_URL` matches the actual origin or allow `http://localhost:5173`. |
| Chat tab says API key missing | Set `OPENAI_API_KEY` in backend environment or provide key in the chat form (depends on configuration). |
| SHAP charts not loading | Confirm backend has access to `models/` and SHAP assets; run from repo root so relative paths resolve. |
| Docker build fails on node modules | Remove `frontend/node_modules` before building or rely on the volume at `/app/node_modules`. |
| Similar offers returning empty results | Try loosening filters (RAM/SSD) or verify `data/` folder is populated with embeddings. |

Need help? Open an issue or contact the team with logs (`backend` prints structured errors around `/api/*` routes).

---

Happy predicting! 🎯
