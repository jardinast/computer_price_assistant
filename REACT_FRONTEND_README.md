# PriceWise - Computer Price Predictor (React Frontend)

A modern, responsive web application for predicting computer prices using machine learning. Built with React + Vite + TailwindCSS and a Flask API backend.

## Features

### 1. Descriptive Analytics (Dashboard)
- **Statistical Overview**: Key metrics like total listings, average price, price range
- **Brand Distribution**: Bar chart showing top computer brands
- **Product Categories**: Pie chart of product types (gaming, multimedia, etc.)
- **Price by Brand**: Average prices comparison across brands
- **RAM Configuration**: Distribution of RAM sizes in the market
- **K-Means Clustering**: Product segmentation into price tiers (Entry-Level, Budget, Mid-Range, Premium, High-End)
- **Interactive Scatter Plot**: Visualize price vs RAM with cluster coloring

### 2. Predictive Analytics (Predictor)
- **Use Case Selection**: Gaming, Work, Creative, Student, General
- **Configuration Form**:
  - Brand selection (optional)
  - Screen size slider
  - CPU brand and family selection (Intel/AMD/Apple)
  - GPU type: Integrated or Dedicated (NVIDIA/AMD/Intel Arc)
  - RAM slider (4GB - 64GB)
  - Storage slider (128GB - 2TB SSD)
  - Refresh rate selection
- **Price Prediction**: ML-powered price estimate with confidence level
- **Feature Importance Chart**: Visual breakdown of price drivers
- **Configuration Summary**: Review your selected specs

### 3. Prescriptive Analytics (Similar Offers)
- **Similarity Search**: Find K-best matching offers based on your criteria
- **Filters**: RAM, SSD, screen size, brand, CPU family, GPU type
- **Results Display**:
  - Product title and specs
  - Real market price
  - Price difference vs prediction
  - Similarity score visualization
- **Price Comparison**: See if offers are above or below predicted price

## Tech Stack

### Frontend
- **React 18** with Hooks
- **Vite** for fast development
- **TailwindCSS** for styling
- **Framer Motion** for animations
- **Recharts** for data visualization
- **Lucide React** for icons
- **Axios** for API calls

### Backend
- **Flask** REST API
- **Flask-CORS** for cross-origin requests
- **Pandas/NumPy** for data processing
- **Scikit-learn** for clustering
- **CatBoost** for price prediction model

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- pip (Python package manager)

### Installation

1. **Install Python dependencies**:
```bash
cd computer-price-predictor
pip install -r requirements.txt
```

2. **Install frontend dependencies**:
```bash
cd frontend
npm install
```

### Running the Application

#### Option 1: Using the run script
```bash
chmod +x run_app.sh
./run_app.sh
```

#### Option 2: Manual startup

**Terminal 1 - Start the Flask API:**
```bash
cd computer-price-predictor/api
python app.py
```
The API will be available at `http://localhost:5000`

**Terminal 2 - Start the React frontend:**
```bash
cd computer-price-predictor/frontend
npm run dev
```
The frontend will be available at `http://localhost:5173`

### Production Deployment (Railway)

1. **Configure env variables**
   - Copy `.env.production.example` to `.env.production` and set `VITE_API_TARGET` to your deployed backend URL (include `/api`, e.g. `https://my-backend.up.railway.app/api`).
   - Railway will read this variable during the build, so also add `VITE_API_TARGET` to the frontend service's variables in the Railway dashboard.
   - On the backend service, set `FRONTEND_URL=https://<your-frontend>.up.railway.app` so CORS allows the deployed SPA.

2. **Build the frontend**
   ```bash
   cd computer-price-predictor/frontend
   npm install
   npm run build
   ```

3. **Deploy a static service**
   - Create a new Railway service pointing to `computer-price-predictor/frontend`.
   - Install command: `npm install`
   - Build command: `npm run build`
   - Start command: `npx serve -s dist -l $PORT` (or any static file server binding to `$PORT`).

4. **Verify connectivity**
   - Open the frontend's Railway URL and ensure API requests hit your backend domain (inspect browser dev tools → Network).

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/descriptive/statistics` | GET | Market statistics |
| `/api/descriptive/clustering` | GET | Clustering data |
| `/api/predictive/options` | GET | Form options |
| `/api/predictive/predict` | POST | Price prediction |
| `/api/predictive/model-info` | GET | Model metadata |
| `/api/prescriptive/similar` | POST | Find similar offers |

## Project Structure

```
computer-price-predictor/
├── api/
│   ├── __init__.py
│   ├── app.py              # Flask API
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx    # Descriptive analytics
│   │   │   ├── Predictor.jsx    # Predictive analytics
│   │   │   └── SimilarOffers.jsx # Prescriptive analytics
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── src/
│   ├── backend_api.py      # Prediction logic
│   ├── benchmark_cache.py  # Data caching
│   ├── defaults.py         # Default values
│   └── modeling.py         # ML utilities
├── models/
│   └── price_model_optimized.pkl
├── data/
│   ├── db_computers_2025_raw.csv
│   ├── db_cpu_raw.csv
│   └── db_gpu_raw.csv
└── run_app.sh
```

## Design Philosophy

### UI/UX Principles
- **Dark Theme**: Easy on the eyes, modern aesthetic
- **Glass Morphism**: Subtle transparency effects for depth
- **Gradient Accents**: Vibrant color gradients for visual interest
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Micro-interactions**: Smooth animations and transitions
- **Progressive Disclosure**: Collapsible sections to reduce cognitive load

### Color Palette
- **Primary**: Sky blue (#0ea5e9)
- **Accent**: Purple (#8b5cf6) to Pink (#d946ef)
- **Surface**: Slate grays (#0f172a to #f8fafc)
- **Success**: Emerald (#10b981)
- **Warning**: Amber (#f59e0b)

### Typography
- **Display**: Outfit (headings)
- **Body**: DM Sans (content)
- **Mono**: JetBrains Mono (prices, data)

## Assignment Requirements Mapping

| Requirement | Implementation |
|-------------|---------------|
| 1A. Statistical Overview | Dashboard page - statistics cards, charts |
| 1B. Clustering/Segmentation | Dashboard page - K-means clustering, scatter plot |
| 2.1 Price Prediction | Predictor page - full form with sliders/dropdowns |
| 2.2 Feature Importance | Predictor page - horizontal bar chart |
| 3. K-best Similar Offers | Similar Offers page - similarity search |

## License

This project was created as part of the DAI Assignment 2.
