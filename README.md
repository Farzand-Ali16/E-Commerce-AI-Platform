# E-Commerce AI Platform

Customer analytics dashboard and customer segment prediction app. The frontend uses React and Vite; the backend uses FastAPI and saved production machine-learning artifacts.

**GitHub repository:** [Farzand-Ali16/E-Commerce-AI-Platform](https://github.com/Farzand-Ali16/E-Commerce-AI-Platform)

> The repository contains the project source code. A public live website has not been deployed yet. The app can be run locally by following the steps below.

## What the project does

- Shows sales, orders, customer counts, average order value, category results, and customer segments.
- Filters dashboard results by date, product category, and city.
- Predicts a segment for one customer from five purchase-history details.
- Uses the saved Core 5 K-means model and preprocessor in `models/`. Starting the app or requesting a prediction does not train or modify the model.

## Requirements

- Python 3.11 or newer
- Node.js and npm

Install Python packages using the same Python interpreter that will run the backend and tests. The saved production artifacts require the versions pinned in `requirements.txt`, including pandas 3.x, NumPy 2.4.4, scikit-learn 1.8.0, and joblib 1.5.3.

## Run locally on Windows

Open PowerShell in the project folder.

### 1. Start the backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --reload
```

Keep this PowerShell window open. The backend API runs at `http://127.0.0.1:8000`; its interactive documentation is at `http://127.0.0.1:8000/docs`.

### 2. Start the frontend

Open a second PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` in your browser. Keep this window open too. The frontend calls the backend at `http://127.0.0.1:8000` by default.

## API base URL

The frontend reads `VITE_API_BASE_URL`. To set it locally, create a file named `frontend/.env` with:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart the Vite server after changing this value. When deploying the frontend and backend separately, set this variable to the public backend URL and allow the deployed frontend origin in the FastAPI CORS settings in `backend/app.py`.

## Dashboard data

The source order data is `data/ecommerce_customer_behavior_dataset_v2.csv`. The dashboard also reads the processed customer features and saved segment assignments from `data/processed/`. Dashboard segment summaries use the saved production assignments; dashboard requests do not retrain K-means.

Date filters include the selected start and end dates. Leaving filters blank shows all available data.

## Customer segment prediction

The prediction screen accepts these five customer details:

1. Total orders
2. Average order value
3. Days since the last order
4. Average days between orders
5. Share of orders in the customer's most common category

The backend validates the values, applies the saved preprocessor, and asks the saved K-means model for the closest segment. A single-order customer does not need an average gap between orders. The prediction screen does not retrain the model.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/segmentation/predict` | Predict one customer's segment |
| `GET /api/segmentation/health` | Check model and preprocessor status |
| `GET /api/dashboard/summary` | Dashboard headline metrics |
| `GET /api/dashboard/sales-trend` | Revenue and orders over time |
| `GET /api/dashboard/category-performance` | Results by product category |
| `GET /api/dashboard/city-performance` | Results by city |
| `GET /api/dashboard/payment-analysis` | Results by payment method |
| `GET /api/dashboard/device-analysis` | Results by device type |
| `GET /api/dashboard/customer-analysis` | Customer purchase analysis |
| `GET /api/dashboard/segments` | Four saved customer segments |
| `GET /api/dashboard/segment/{cluster_id}` | Details for one segment |
| `GET /api/dashboard/filters` | Available dashboard filter values |
| `GET /api/dashboard/health` | Check dashboard data status |

Dashboard endpoints support optional date, category, city, payment method, device type, and segment filters where applicable. Invalid filters return a validation error.

## Tests

From the project root, with the Python environment active:

```powershell
python -m pytest tests -q
```

## Project structure

```text
backend/       FastAPI application, API routes, schemas, and services
data/          Source dataset and processed customer/segment data
frontend/      React + Vite user interface
models/        Saved model, preprocessor, and segment metadata
reports/       Analysis reports and figures
scripts/       Environment diagnostics
src/           Data, feature, preprocessing, and model pipeline code
tests/         Backend API and segmentation tests
requirements.txt
```
