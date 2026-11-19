# E-Commerce AI Analytics & Customer Intelligence Platform

## Environment (important)

The saved production artifacts need **pandas 3.x** (the preprocessor stores a pandas 3 string dtype). Python 3.11+ is required.
Exact versions are pinned in `requirements.txt` (pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, joblib 1.5.3).

```bash
python -m pip install -r requirements.txt     # use the SAME python that runs the app and pytest
python scripts/diagnose_environment.py        # read-only check: versions, file hashes, loads both artifacts
```

## Stage 6: segment prediction API

```bash
pip install -r requirements.txt
python -m uvicorn backend.app:app --reload      # run from the project root
# docs: http://127.0.0.1:8000/docs
python -m pytest tests -v                        # tests
python tests/test_segmentation_service.py        # demo table + tests
```

## Stage 7: dashboard API (`/api/dashboard`)

Read-only GET endpoints that calculate everything from the real CSV files (loaded once at start-up, never retrained):

| Endpoint | Purpose |
|---|---|
| `GET /api/dashboard/summary` | KPIs: revenue, orders, customers, AOV, rating, quantity |
| `GET /api/dashboard/sales-trend?granularity=day\|week\|month` | Revenue and orders over time |
| `GET /api/dashboard/category-performance` | Revenue, orders, AOV by category |
| `GET /api/dashboard/city-performance` | Revenue and orders by city |
| `GET /api/dashboard/payment-analysis` | Revenue and orders by payment method |
| `GET /api/dashboard/device-analysis` | Revenue, orders, engagement by device |
| `GET /api/dashboard/customer-analysis` | Orders per customer, spending, ratings, recency, new vs returning |
| `GET /api/dashboard/segments` | The 4 production segments side by side |
| `GET /api/dashboard/segment/{cluster_id}` | One segment in detail (404 for unknown id) |
| `GET /api/dashboard/filters` | Values for the filter dropdowns |
| `GET /api/dashboard/health` | Data loaded? |

Optional filters on every endpoint: `start_date`, `end_date` (YYYY-MM-DD, inclusive), `category`, `city`,
`payment_method`, `device_type` (case-insensitive) and `segment` (0-3; not on the two segment endpoints).
Customer numbers are calculated from the selected orders. Invalid values return 422 with the allowed values.

## Stage 8: React + Vite analytics dashboard

Stage 8 adds a browser dashboard that reads the Stage 7 API and a customer segment prediction screen. It uses the
saved production model through the API; the frontend never trains or changes model artifacts.

### Backend setup and run

Use Python 3.11 or newer. From the project root, create an environment, install the pinned backend requirements,
and start FastAPI:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive API documentation is at
`http://127.0.0.1:8000/docs`. On startup, it loads the existing data and production artifacts from `data/` and
`models/`.

### Frontend setup and run

Open a second terminal at `frontend/`. Vite 6 requires Node.js 18+, 20+, or 22+ and npm.

```bash
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://127.0.0.1:5173`). To create and serve a production
build, run `npm run build` and then `npm run preview` from `frontend/`.

### API base URL configuration

The frontend reads `VITE_API_BASE_URL`. For local development, copy `frontend/.env.example` to `frontend/.env`
and use `http://127.0.0.1:8000`. Set it to the deployed backend origin when using another server. The API allows
the default Vite development origins `http://localhost:5173` and `http://127.0.0.1:5173`; add your deployed
frontend origin to the FastAPI CORS configuration in `backend/app.py` when hosting the two separately.

The React client sends JSON `fetch` requests directly to this base URL. Dashboard views call the read-only
`/api/dashboard/*` endpoints; segment predictions use `POST /api/segmentation/predict`. FastAPI serves the API
on port `8000` and allows both default Vite development origins in its CORS configuration.

### Stage 8 features

- Sales overview with revenue, order, customer, average order value, rating, and quantity KPIs.
- Monthly revenue trend, category performance, customer repeat-rate summary, and the four saved customer segments.
- Date, category, and city filters applied to dashboard analytics.
- Segment detail view with customer profile, purchase mix, and suggested actions.
- Customer prediction form for the five production features, with assignment strength, nearest alternative, model
  profile, and validation/extrapolation warnings.
- Backend availability indicator and readable loading, empty, and error states.

### Project structure

```text
backend/       FastAPI app, routes, schemas, and services
data/          Source dataset and processed analytics / segment data
frontend/      React + Vite dashboard
models/        Saved production model, preprocessor, and segment metadata
reports/       Existing analysis reports and figures
scripts/       Environment diagnostics
src/           Data, feature, preprocessing, and model pipeline source
tests/         Backend API and segmentation tests
requirements.txt
```

### Run backend tests

With the backend environment active, from the project root run:

```bash
python -m pytest tests -q
```
