"""
FastAPI Backend for Computer Price Predictor

Provides REST API endpoints for:
- Price prediction
- Market statistics
- Clustering/segmentation
- Similar laptop search
- Chat advisor (OpenAI integration)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import sys
import os
import math
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend_api import (
    predecir_precio,
    explicar_prediccion,
    obtener_campos_disponibles,
    get_model_info,
)
from src.analytics import (
    get_market_statistics,
    get_price_by_category,
    get_clustering_data,
    get_cluster_profiles,
    find_similar_laptops,
    load_computers_data,
    get_descriptive_statistics,
    get_clustering_summary,
)
from src.llm_advisor import ChatAdvisor


def _safe_float(value):
    """Convert value to float if possible, ignoring NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _infer_cpu_brand_from_family(cpu_family: Optional[str]) -> Optional[str]:
    """Infer CPU brand from cpu family string."""
    if not cpu_family:
        return None
    family = cpu_family.lower()
    if 'ryzen' in family or 'amd' in family:
        return 'amd'
    if 'core' in family or 'intel' in family or family.startswith('i'):
        return 'intel'
    if family.startswith('m'):
        return 'apple'
    if 'snapdragon' in family or 'qualcomm' in family:
        return 'qualcomm'
    return None

# Initialize FastAPI app
app = FastAPI(
    title="Computer Price Predictor API",
    description="API for laptop price prediction, market analysis, and recommendations",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    brand: Optional[str] = Field(None, description="Laptop brand")
    cpu_brand: Optional[str] = Field("intel", description="CPU manufacturer")
    cpu_family: str = Field(..., description="CPU family (e.g., 'core i7', 'ryzen 7')")
    ram_gb: int = Field(16, ge=4, le=128, description="RAM in GB")
    ssd_gb: int = Field(512, ge=128, le=4096, description="Storage in GB")
    gpu_brand: Optional[str] = Field(None, description="GPU manufacturer")
    gpu_series: Optional[str] = Field("integrated", description="GPU model")
    gpu_is_integrated: bool = Field(True, description="Is GPU integrated?")
    screen_size: Optional[float] = Field(None, ge=11.0, le=18.0, description="Screen size in inches")
    refresh_rate: Optional[int] = Field(None, description="Refresh rate in Hz")
    weight_kg: Optional[float] = Field(None, description="Weight in kg")
    use_case: str = Field("general", description="Use case: gaming, work, creative, student, general")


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    predicted_price: float
    price_min: float
    price_max: float
    confidence: str
    top_features: List[Dict[str, Any]]
    explanation: str


class SimilarLaptopsRequest(BaseModel):
    """Request for finding similar laptops."""
    price_target: float = Field(..., description="Target price in EUR")
    ram_gb: int = Field(16, description="RAM in GB")
    ssd_gb: int = Field(512, description="Storage in GB")
    screen_size: float = Field(15.6, description="Screen size in inches")
    brand: Optional[str] = Field(None, description="Preferred brand")
    cpu_brand: Optional[str] = Field(None, description="Preferred CPU brand")
    num_results: int = Field(5, ge=1, le=20, description="Number of results")


class ChatRequest(BaseModel):
    """Request for chat advisor."""
    message: str = Field(..., description="User message")
    api_key: str = Field(..., description="OpenAI API key")
    conversation_history: List[Dict[str, str]] = Field(default=[], description="Previous messages")


class ChatResponse(BaseModel):
    """Response from chat advisor."""
    response: str
    conversation_history: List[Dict[str, str]]


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Computer Price Predictor API"}


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    try:
        model_info = get_model_info()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_info": model_info
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e)
        }


# -----------------------------------------------------------------------------
# PREDICTIVE ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict laptop price based on specifications.

    Returns predicted price with confidence interval and feature importance.
    """
    try:
        # Build input dict for backend
        inputs = {
            '_brand': request.brand.lower() if request.brand else 'asus',
            'cpu_brand': request.cpu_brand.lower() if request.cpu_brand else 'intel',
            'cpu_family': request.cpu_family.lower(),
            '_ram_gb': request.ram_gb,
            '_ssd_gb': request.ssd_gb,
            'gpu_brand': request.gpu_brand.lower() if request.gpu_brand else 'intel',
            'gpu_series': request.gpu_series.lower() if request.gpu_series else 'integrated',
            'gpu_is_integrated': request.gpu_is_integrated,
        }

        if request.screen_size:
            inputs['_tamano_pantalla_pulgadas'] = request.screen_size
        if request.refresh_rate:
            inputs['_tasa_refresco_hz'] = request.refresh_rate
        if request.weight_kg:
            inputs['_peso_kg'] = request.weight_kg

        # Get prediction with explanation
        result = explicar_prediccion(inputs, use_case=request.use_case)

        return PredictionResponse(
            predicted_price=result['prediccion'],
            price_min=result['prediccion_min'],
            price_max=result['prediccion_max'],
            confidence=result['confidence'],
            top_features=result['top_features'][:10],
            explanation=result['explanation_text']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/form-options")
async def get_form_options():
    """Get available options for form dropdowns."""
    try:
        options = obtener_campos_disponibles()
        return options
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictive/options")
async def get_predictive_options():
    """Alias endpoint for React frontend to fetch prediction options."""
    try:
        return obtener_campos_disponibles()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predictive/predict")
async def predictive_predict(payload: Dict[str, Any]):
    """Predict price using payload structure expected by the React frontend."""
    if not payload:
        raise HTTPException(status_code=400, detail="No data provided")

    use_case = payload.get('use_case', 'general')
    inputs = {k: v for k, v in payload.items() if k not in {'use_case', 'k'} and v not in (None, '')}

    try:
        result = explicar_prediccion(inputs, use_case=use_case)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top_features = []
    for feature in result.get('top_features', []):
        importance = feature.get('importance_pct', 0)
        importance_val = _safe_float(importance)
        top_features.append({
            'feature': feature.get('feature'),
            'readable_name': feature.get('readable_name', feature.get('feature')),
            'importance_pct': importance_val if importance_val is not None else 0,
        })

    return {
        'predicted_price': _safe_float(result.get('prediccion')) or 0,
        'price_range': {
            'min': _safe_float(result.get('prediccion_min')) or 0,
            'max': _safe_float(result.get('prediccion_max')) or 0,
        },
        'confidence': result.get('confidence', 'unknown'),
        'top_features': top_features,
        'explanation': result.get('explanation_text', ''),
    }


@app.get("/api/predictive/model-info")
async def predictive_model_info():
    """Expose model metadata for frontend diagnostics."""
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictive/use-case-profiles")
async def predictive_use_case_profiles():
    """Provide use case profiles expected by the frontend."""
    profiles = {}
    for name, profile in USE_CASE_PROFILES.items():
        profiles[name] = {
            'ram_gb': profile.get('_ram_gb'),
            'ssd_gb': profile.get('_ssd_gb'),
            'screen_inches': profile.get('_tamano_pantalla_pulgadas'),
            'refresh_rate': profile.get('_tasa_refresco_hz'),
            'cpu_family': profile.get('cpu_family'),
            'gpu_series': profile.get('gpu_series'),
            'gpu_integrated': profile.get('gpu_is_integrated'),
        }
    return profiles


# -----------------------------------------------------------------------------
# DESCRIPTIVE ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/statistics")
async def get_statistics():
    """Get market statistics overview."""
    try:
        stats = get_market_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/descriptive/statistics")
async def get_descriptive_stats():
    """Get detailed descriptive statistics for the dashboard."""
    try:
        return get_descriptive_statistics()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/descriptive/clustering")
async def get_descriptive_clustering():
    """Get clustering data formatted for the dashboard."""
    try:
        return get_clustering_summary()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics/by-category")
async def get_stats_by_category(category: str = Query("brand", enum=["brand", "cpu", "gpu"])):
    """Get price statistics grouped by category."""
    try:
        result = get_price_by_category(category)
        return result.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clusters")
async def get_clusters():
    """Get clustering/segmentation data."""
    try:
        df, X_pca, clusters = get_clustering_data()
        profiles = get_cluster_profiles()

        # Prepare PCA data for visualization
        pca_data = []
        if len(X_pca) > 0:
            for i in range(min(1000, len(X_pca))):  # Limit for performance
                pca_data.append({
                    'x': float(X_pca[i, 0]),
                    'y': float(X_pca[i, 1]),
                    'cluster': int(clusters[i])
                })

        return {
            'profiles': profiles,
            'pca_data': pca_data,
            'total_clustered': len(clusters) if len(clusters) > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# PRESCRIPTIVE ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/prescriptive/similar")
async def prescriptive_similar(payload: Dict[str, Any]):
    """Find similar offers using the simplified payload expected by the frontend."""
    if not payload:
        raise HTTPException(status_code=400, detail="No data provided")

    try:
        k = int(payload.get('k', 10))
    except (TypeError, ValueError):
        k = 10
    k = max(1, min(k, 25))

    use_case = payload.get('use_case', 'general')
    prediction_inputs = {k: v for k, v in payload.items() if k not in {'k'}}

    predicted_price = None
    try:
        prediction = explicar_prediccion(
            {k: v for k, v in prediction_inputs.items() if k != 'use_case'},
            use_case=use_case
        )
        predicted_price = _safe_float(prediction.get('prediccion'))
    except Exception:
        predicted_price = None

    specs = {}
    price_target = predicted_price or _safe_float(payload.get('price_target'))
    if price_target is not None:
        specs['price_target'] = price_target

    ram_value = _safe_float(payload.get('_ram_gb'))
    if ram_value is not None:
        specs['ram_gb'] = ram_value

    ssd_value = _safe_float(payload.get('_ssd_gb'))
    if ssd_value is not None:
        specs['ssd_gb'] = ssd_value

    screen_value = _safe_float(payload.get('_tamano_pantalla_pulgadas'))
    if screen_value is not None:
        specs['screen_size'] = screen_value

    brand_value = payload.get('_brand')
    if brand_value:
        specs['brand'] = brand_value

    cpu_brand = payload.get('cpu_brand') or _infer_cpu_brand_from_family(payload.get('cpu_family'))
    if cpu_brand:
        specs['cpu_brand'] = cpu_brand

    results = find_similar_laptops(specs, k=k)

    offers = []
    if not results.empty:
        for _, row in results.iterrows():
            title = row.get('Título', 'Unknown')
            if isinstance(title, float) and math.isnan(title):
                title = 'Unknown'

            brand_name = row.get('_brand', 'Unknown')
            if isinstance(brand_name, float) and math.isnan(brand_name):
                brand_name = 'Unknown'

            real_price = _safe_float(row.get('_precio_medio'))
            product_type = row.get('Tipo de producto')
            if isinstance(product_type, float) and math.isnan(product_type):
                product_type = None
            elif product_type is not None and not isinstance(product_type, str):
                product_type = str(product_type)

            price_diff = None
            if predicted_price is not None and real_price is not None:
                price_diff = real_price - predicted_price

            offers.append({
                'title': title,
                'real_price': real_price,
                'price_min': _safe_float(row.get('_precio_min')),
                'price_max': _safe_float(row.get('_precio_max')),
                'brand': brand_name,
                'product_type': product_type,
                'ram_gb': _safe_float(row.get('_ram_gb')),
                'ssd_gb': _safe_float(row.get('_ssd_gb')),
                'screen_inches': _safe_float(row.get('_screen_size')),
                'distance': _safe_float(row.get('_similarity_distance')) or 0,
                'price_difference': price_diff,
            })

    return {
        'predicted_price': predicted_price,
        'target_features': {
            'ram_gb': _safe_float(payload.get('_ram_gb')),
            'ssd_gb': _safe_float(payload.get('_ssd_gb')),
            'screen_inches': _safe_float(payload.get('_tamano_pantalla_pulgadas')),
            'brand': payload.get('_brand'),
        },
        'similar_offers': offers,
    }

@app.post("/api/similar")
async def find_similar(request: SimilarLaptopsRequest):
    """Find k-best similar laptops based on specifications."""
    try:
        specs = {
            'price_target': request.price_target,
            'ram_gb': request.ram_gb,
            'ssd_gb': request.ssd_gb,
            'screen_size': request.screen_size,
        }

        if request.brand:
            specs['brand'] = request.brand
        if request.cpu_brand:
            specs['cpu_brand'] = request.cpu_brand

        results = find_similar_laptops(specs, k=request.num_results)

        if results.empty:
            return {'laptops': [], 'count': 0}

        # Convert to list of dicts
        laptops = []
        for _, row in results.iterrows():
            laptops.append({
                'rank': int(row.get('_similarity_rank', 0)),
                'title': row.get('Título', 'Unknown'),
                'price': float(row.get('_precio_medio', 0)),
                'price_min': float(row.get('_precio_min', 0)),
                'price_max': float(row.get('_precio_max', 0)),
                'brand': row.get('_brand', 'Unknown'),
                'ram_gb': float(row.get('_ram_gb', 0)) if row.get('_ram_gb') else None,
                'ssd_gb': float(row.get('_ssd_gb', 0)) if row.get('_ssd_gb') else None,
                'screen_size': float(row.get('_screen_size', 0)) if row.get('_screen_size') else None,
                'cpu_brand': row.get('_cpu_brand', 'Unknown'),
                'gpu_brand': row.get('_gpu_brand', 'Unknown'),
                'distance': float(row.get('_similarity_distance', 0)),
            })

        return {'laptops': laptops, 'count': len(laptops)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# CHAT ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI laptop advisor."""
    try:
        # Initialize advisor with API key
        advisor = ChatAdvisor(api_key=request.api_key)

        # Restore conversation history
        advisor.messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in request.conversation_history
        ]

        # Get response
        response = advisor.chat(request.message)

        # Build updated history
        new_history = request.conversation_history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ]

        return ChatResponse(
            response=response,
            conversation_history=new_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/greeting")
async def get_chat_greeting():
    """Get the initial greeting for the chat advisor."""
    return {
        "greeting": """Hi! I'm your laptop advisor. I'm here to help you find the perfect computer for your needs.

**What will you mainly use this laptop for?**

- 🎮 **Gaming** - Playing video games
- 💼 **Work** - Business and productivity
- 🎨 **Creative** - Video editing, design, 3D work
- 📚 **Student** - School and studying
- 🏠 **General** - Everyday tasks

Just tell me about your needs, and I'll help you find the right specs and price range!"""
    }


# -----------------------------------------------------------------------------
# STATIC FILES (for production - serve React build)
# -----------------------------------------------------------------------------

# Serve static files if the static directory exists (production build)
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/assets", StaticFiles(directory=static_path / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        # Don't serve index.html for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        # Try to serve the requested file
        file_path = static_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Fall back to index.html for SPA routing
        return FileResponse(static_path / "index.html")


# -----------------------------------------------------------------------------
# RUN SERVER
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
