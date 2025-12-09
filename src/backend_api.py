"""
backend_api.py - API Wrapper for Streamlit Frontend

This module provides a thin API layer that the Streamlit frontend
can use to get price predictions and explanations.

Usage:
------
>>> from src.backend_api import predecir_precio, explicar_prediccion
>>>
>>> inputs = {
...     '_ram_gb': 16,
...     '_ssd_gb': 512,
...     'cpu_family': 'core i7',
...     'gpu_series': 'rtx 4060',
...     # ... other fields
... }
>>>
>>> precio_predicho = predecir_precio(inputs)
>>> print(f"Predicted price: {precio_predicho:.2f} EUR")
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib

# Import local modules
from .defaults import (
    fill_missing_with_defaults,
    get_cpu_benchmark_defaults,
    get_gpu_benchmark_defaults,
    get_confidence_level,
    USE_CASE_PROFILES
)
from .benchmark_cache import (
    get_cpu_data,
    get_gpu_data,
    get_brand_options,
    get_cpu_family_options,
    get_gpu_series_options,
    get_resolution_options,
    get_refresh_rate_options,
    get_ram_options,
    get_storage_options,
)

# Path to models
MODELS_DIR = Path(__file__).parent.parent / 'models'
DEFAULT_MODEL_PATH = MODELS_DIR / 'price_model_optimized.pkl'

# Global cache for loaded model
_modelo_cargado = None


def _cargar_modelo_si_necesario(model_path: str = None):
    """
    Load the model if not already loaded.

    Uses a global cache to avoid reloading on every prediction.

    Parameters
    ----------
    model_path : str, optional
        Path to model file. Defaults to price_model_optimized.pkl

    Returns
    -------
    dict
        Dictionary with 'model' and 'metadata' keys
    """
    global _modelo_cargado

    if _modelo_cargado is None:
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not path.exists():
            raise RuntimeError(f"Model file not found at {path}")

        saved = joblib.load(path)
        _modelo_cargado = {
            'model': saved['model'],
            'metadata': saved.get('metadata', {}),
        }

        # Extract feature metadata and fall back to CatBoost info when needed
        meta = _modelo_cargado['metadata']
        feature_cols = meta.get('feature_cols', [])
        if hasattr(_modelo_cargado['model'], 'feature_names_'):
            names = list(getattr(_modelo_cargado['model'], 'feature_names_'))
            if names:
                feature_cols = names
        numeric_cols = meta.get('numeric_cols', [])
        categorical_cols = meta.get('categorical_cols', [])

        if feature_cols:
            try:
                cat_indices = _modelo_cargado['model'].get_cat_feature_indices()
                categorical_cols = [
                    feature_cols[i]
                    for i in cat_indices
                    if i < len(feature_cols)
                ]
            except Exception:
                # If CatBoost introspection fails, keep metadata-provided list
                pass

        _modelo_cargado['feature_cols'] = feature_cols
        _modelo_cargado['numeric_cols'] = numeric_cols
        _modelo_cargado['categorical_cols'] = categorical_cols

    return _modelo_cargado


def _prepare_features(campos_dict: Dict[str, Any], use_case: str = 'general') -> pd.DataFrame:
    """
    Prepare feature DataFrame from user inputs.

    Parameters
    ----------
    campos_dict : Dict[str, Any]
        User-provided input values
    use_case : str
        Use case for filling defaults

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all required features
    """
    # Fill missing values with defaults
    filled = fill_missing_with_defaults(campos_dict, use_case)

    # Add CPU benchmark defaults if not provided
    cpu_family = filled.get('cpu_family', 'core i5')
    if '_cpu_mark' not in filled or filled.get('_cpu_mark') is None:
        cpu_benchmarks = get_cpu_benchmark_defaults(cpu_family)
        for key, value in cpu_benchmarks.items():
            if key not in filled or filled[key] is None:
                filled[key] = value

    # Add GPU benchmark defaults if not provided
    gpu_series = filled.get('gpu_series', 'integrated')
    is_integrated = filled.get('gpu_is_integrated', True)
    if '_gpu_mark' not in filled or filled.get('_gpu_mark') is None:
        gpu_benchmarks = get_gpu_benchmark_defaults(gpu_series, is_integrated)
        for key, value in gpu_benchmarks.items():
            if key not in filled or filled[key] is None:
                filled[key] = value

    # Create DataFrame
    df = pd.DataFrame([filled])

    return df


def predecir_precio(campos_dict: Dict[str, Any],
                    use_case: str = 'general',
                    model_path: str = None) -> float:
    """
    Predict the price for a computer based on user inputs.

    Parameters
    ----------
    campos_dict : Dict[str, Any]
        Dictionary mapping feature names to values.
        Key features:
        - _ram_gb: RAM size in GB
        - _ssd_gb: Storage size in GB
        - cpu_family: CPU family (e.g., 'core i7', 'ryzen 7')
        - gpu_series: GPU series (e.g., 'rtx 4060', 'integrated')
        - _brand: Laptop brand
        - _tamano_pantalla_pulgadas: Screen size in inches
        - gpu_is_integrated: Boolean for integrated GPU

    use_case : str
        Use case for filling missing values: 'gaming', 'work', 'creative', 'student', 'general'

    model_path : str, optional
        Path to model file

    Returns
    -------
    float
        Predicted price in EUR

    Examples
    --------
    >>> inputs = {
    ...     '_ram_gb': 16,
    ...     '_ssd_gb': 512,
    ...     'cpu_family': 'core i7',
    ...     'gpu_series': 'rtx 4060',
    ...     '_brand': 'ASUS',
    ... }
    >>> precio = predecir_precio(inputs, use_case='gaming')
    >>> print(f"Estimated price: {precio:.2f} EUR")
    """
    # Load model
    modelo_info = _cargar_modelo_si_necesario(model_path)
    model = modelo_info['model']
    feature_cols = modelo_info['feature_cols']
    categorical_cols = modelo_info['categorical_cols']

    # Prepare features
    df = _prepare_features(campos_dict, use_case)

    # Select only model features that exist in our DataFrame
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]

    # Add missing columns with NaN
    for col in missing_cols:
        df[col] = np.nan

    # Ensure categorical columns are strings (after adding missing columns)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', 'missing')

    # Reorder to match model expectations
    X = df[feature_cols]

    # Make prediction
    prediction = model.predict(X)[0]

    # Ensure non-negative
    return float(max(0, prediction))


def explicar_prediccion(campos_dict: Dict[str, Any],
                        use_case: str = 'general',
                        model_path: str = None) -> Dict[str, Any]:
    """
    Explain the price prediction with both global and local (SHAP) feature importances.

    Parameters
    ----------
    campos_dict : Dict[str, Any]
        Same as predecir_precio
    use_case : str
        Use case for filling missing values
    model_path : str, optional
        Path to model file

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - prediccion: Predicted price
        - prediccion_min: Lower bound (price - 20%)
        - prediccion_max: Upper bound (price + 20%)
        - confidence: 'high', 'medium', or 'low'
        - top_features: List of top contributing features (global)
        - shap_features: List of SHAP values for this specific prediction (local)
        - feature_importances: Dict of all feature importances
    """
    # Load model
    modelo_info = _cargar_modelo_si_necesario(model_path)
    model = modelo_info['model']
    feature_cols = modelo_info['feature_cols']
    
    # Get use case profile
    profile = USE_CASE_PROFILES.get(use_case, USE_CASE_PROFILES['general'])
    
    # Prepare input data (same logic as predecir_precio)
    merged = {**profile, **campos_dict}
    input_data = {col: merged.get(col, None) for col in feature_cols}
    X = pd.DataFrame([input_data])
    
    # Make prediction
    precio = float(max(0, model.predict(X)[0]))

    # Create readable feature names
    def readable_name(feature_name: str) -> str:
        mapping = {
            '_cpu_mark': 'CPU Performance',
            '_cpu_rank': 'CPU Ranking',
            '_cpu_value': 'CPU Value Score',
            '_cpu_price_usd': 'CPU Price',
            '_gpu_mark': 'GPU Performance',
            '_gpu_rank': 'GPU Ranking',
            '_gpu_value': 'GPU Value Score',
            '_gpu_price_usd': 'GPU Price',
            '_ram_gb': 'RAM Size',
            '_ssd_gb': 'Storage Size',
            '_tamano_pantalla_pulgadas': 'Screen Size',
            '_peso_kg': 'Weight',
            '_resolucion_pixeles': 'Screen Resolution',
            '_tasa_refresco_hz': 'Refresh Rate',
            '_gpu_memory_gb': 'GPU Memory',
            '_cpu_cores': 'CPU Cores',
            '_brand': 'Brand',
            '_serie': 'Product Series',
            'Serie': 'Product Series',
            'cpu_brand': 'CPU Brand',
            'cpu_family': 'CPU Family',
            'gpu_brand': 'GPU Brand',
            'gpu_series': 'GPU Series',
            'gpu_is_integrated': 'Integrated GPU',
            '_tiene_wifi': 'WiFi',
            '_tiene_bluetooth': 'Bluetooth',
            '_tiene_webcam': 'Webcam',
        }
        return mapping.get(feature_name, feature_name.replace('_', ' ').title())

    # Get GLOBAL feature importances from CatBoost
    try:
        importances = model.get_feature_importance()
        feature_importances = dict(zip(feature_cols, importances))
    except Exception:
        feature_importances = {col: 1.0 / len(feature_cols) for col in feature_cols}

    # Sort by importance (global)
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Get top features (global)
    total_importance = sum(abs(v) for v in feature_importances.values())
    top_features = []
    for feature_name, importance in sorted_features[:10]:
        importance_pct = (abs(importance) / total_importance * 100) if total_importance > 0 else 0
        top_features.append({
            'feature': feature_name,
            'readable_name': readable_name(feature_name),
            'importance': importance,
            'importance_pct': importance_pct,
        })

    # Get LOCAL SHAP values for this specific prediction
    shap_features = []
    try:
        from catboost import Pool
        
        # Get categorical feature names from model
        cat_features = modelo_info.get('cat_features', [])
        
        # Create a Pool object for SHAP calculation
        pool = Pool(X, cat_features=cat_features if cat_features else None)
        
        # CatBoost's get_feature_importance with ShapValues
        shap_values = model.get_feature_importance(
            data=pool,
            type='ShapValues'
        )
        # ShapValues returns array of shape (n_samples, n_features + 1)
        # Last column is the base value (expected value)
        if len(shap_values) > 0:
            shap_row = shap_values[0]  # First (and only) sample
            base_value = shap_row[-1]  # Last element is base value
            feature_shaps = shap_row[:-1]  # All but last are feature contributions
            
            # Create list of (feature, shap_value) and sort by absolute value
            shap_pairs = list(zip(feature_cols, feature_shaps))
            shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
            
            for feature_name, shap_value in shap_pairs_sorted[:10]:
                shap_features.append({
                    'feature': feature_name,
                    'readable_name': readable_name(feature_name),
                    'shap_value': float(shap_value),  # Actual € contribution
                    'direction': 'positive' if shap_value > 0 else 'negative',
                })
    except Exception as e:
        # Fallback if SHAP not available - use global importance scaled by price
        import traceback
        print(f"SHAP calculation failed: {e}")
        print(traceback.format_exc())
    
    # If SHAP failed or returned empty, use fallback
    if not shap_features:
        for feat in top_features[:10]:
            contribution = precio * (feat['importance_pct'] / 100)
            shap_features.append({
                'feature': feat['feature'],
                'readable_name': feat['readable_name'],
                'shap_value': contribution,
                'direction': 'positive',
            })

    # Calculate confidence
    confidence = get_confidence_level(campos_dict)

    # Generate explanation text from SHAP values
    top_3_shap = shap_features[:3]
    explanation_parts = []
    for feat in top_3_shap:
        sign = '+' if feat['shap_value'] > 0 else ''
        explanation_parts.append(f"{feat['readable_name']} ({sign}€{feat['shap_value']:.0f})")

    explanation_text = f"Price influenced by: {', '.join(explanation_parts)}"

    return {
        'prediccion': precio,
        'prediccion_min': precio * 0.80,  # -20% based on MAPE
        'prediccion_max': precio * 1.20,  # +20% based on MAPE
        'confidence': confidence,
        'top_features': top_features,  # Global feature importance
        'shap_features': shap_features,  # Local SHAP values for THIS prediction
        'feature_importances': feature_importances,
        'explanation_text': explanation_text,
    }


def obtener_campos_disponibles() -> Dict[str, Any]:
    """
    Get information about available input fields for the frontend.

    Returns
    -------
    Dict[str, Any]
        Dictionary with field metadata for building UI
    """
    return {
        'use_cases': list(USE_CASE_PROFILES.keys()),
        'brands': get_brand_options(),
        'cpu_families': {
            'Intel': get_cpu_family_options('Intel'),
            'AMD': get_cpu_family_options('AMD'),
            'Apple': get_cpu_family_options('Apple'),
            'Qualcomm': get_cpu_family_options('Qualcomm'),
        },
        'gpu_options': {
            'Integrated': get_gpu_series_options(integrated=True),
            'NVIDIA': get_gpu_series_options('NVIDIA'),
            'AMD': get_gpu_series_options('AMD'),
            'Intel Arc': get_gpu_series_options('Intel Arc'),
        },
        'resolutions': get_resolution_options(),
        'refresh_rates': get_refresh_rate_options(),
        'ram_options': get_ram_options(),
        'storage_options': get_storage_options(),
        'numeric_ranges': {
            '_ram_gb': (4, 128),
            '_ssd_gb': (128, 4096),
            '_tamano_pantalla_pulgadas': (11.0, 18.0),
            '_peso_kg': (0.5, 4.5),
            '_tasa_refresco_hz': (60, 360),
            '_gpu_memory_gb': (0, 24),
            '_cpu_cores': (2, 24),
        },
        'simple_mode_fields': [
            'use_case', '_brand', 'cpu_family', '_ram_gb', '_ssd_gb', 'gpu_series'
        ],
        'required_fields': ['cpu_family', '_ram_gb'],
    }


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns
    -------
    Dict[str, Any]
        Model metadata and statistics
    """
    modelo_info = _cargar_modelo_si_necesario()
    metadata = modelo_info['metadata']

    return {
        'feature_count': len(modelo_info['feature_cols']),
        'numeric_features': len(modelo_info['numeric_cols']),
        'categorical_features': len(modelo_info['categorical_cols']),
        'metrics': metadata.get('metrics', {}),
        'training_date': metadata.get('training_date', 'Unknown'),
        'price_range': {
            'min': metadata.get('price_min', 0),
            'max': metadata.get('price_max', 3605),
        }
    }


def reload_model(model_path: str = None):
    """
    Force reload the model from disk.

    Parameters
    ----------
    model_path : str, optional
        Path to model file
    """
    global _modelo_cargado
    _modelo_cargado = None
    _cargar_modelo_si_necesario(model_path)
