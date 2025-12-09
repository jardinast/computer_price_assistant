"""
Flask API Backend for Computer Price Prediction

Provides REST endpoints for:
- Descriptive Analytics (statistics, clustering)
- Predictive Analytics (price prediction, feature importance)
- Prescriptive Analytics (k-best similar offers)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend_api import (
    predecir_precio,
    explicar_prediccion,
    obtener_campos_disponibles,
    get_model_info,
)
from src.benchmark_cache import (
    get_brand_options,
    get_cpu_family_options,
    get_gpu_series_options,
    get_resolution_options,
    get_refresh_rate_options,
    get_ram_options,
    get_storage_options,
)
from src.defaults import USE_CASE_PROFILES

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000'])

# Data paths
DATA_DIR = Path(__file__).parent.parent / 'data'
MODELS_DIR = Path(__file__).parent.parent / 'models'

# Cache for loaded data
_cached_data = {}


def get_computers_data():
    """Load and cache the computers dataset."""
    if 'computers' not in _cached_data:
        df = pd.read_csv(DATA_DIR / 'db_computers_2025_raw.csv', encoding='utf-8-sig', index_col=0)
        df = df.reset_index(drop=True)
        
        # Extract price numeric
        import re
        def extract_price(precio_str):
            if pd.isna(precio_str):
                return np.nan
            pattern = r'([\d.]+,\d{2})'
            matches = re.findall(pattern, str(precio_str))
            if not matches:
                return np.nan
            precios = []
            for m in matches:
                num_str = m.replace('.', '').replace(',', '.')
                precios.append(float(num_str))
            return np.mean(precios) if precios else np.nan
        
        df['_precio_num'] = df['Precio_Rango'].apply(extract_price)
        
        # Extract RAM
        def extract_ram(text):
            if pd.isna(text):
                return np.nan
            match = re.search(r'(\d+)\s*GB', str(text).upper())
            return float(match.group(1)) if match else np.nan
        
        df['_ram_gb'] = df['RAM_Memoria RAM'].apply(extract_ram) if 'RAM_Memoria RAM' in df.columns else np.nan
        
        # Extract SSD
        def extract_ssd(text):
            if pd.isna(text):
                return np.nan
            text = str(text).upper().replace('.', '')
            match_tb = re.search(r'(\d+)\s*TB', text)
            if match_tb:
                return float(match_tb.group(1)) * 1000
            match_gb = re.search(r'(\d+)\s*GB', text)
            return float(match_gb.group(1)) if match_gb else np.nan
        
        df['_ssd_gb'] = df['Disco duro_Capacidad de memoria SSD'].apply(extract_ssd) if 'Disco duro_Capacidad de memoria SSD' in df.columns else np.nan
        
        # Extract screen size
        def extract_screen(text):
            if pd.isna(text):
                return np.nan
            match = re.search(r'([\d,\.]+)', str(text))
            if match:
                return float(match.group(1).replace(',', '.'))
            return np.nan
        
        df['_screen_inches'] = df['Pantalla_Tamaño de la pantalla'].apply(extract_screen) if 'Pantalla_Tamaño de la pantalla' in df.columns else np.nan
        
        # Extract brand
        def extract_brand(titulo):
            if pd.isna(titulo):
                return 'Other'
            brands = ['Apple', 'ASUS', 'Lenovo', 'HP', 'Dell', 'Acer', 'MSI', 
                     'Samsung', 'Microsoft', 'Razer', 'LG', 'Huawei', 'Gigabyte', 'Toshiba', 'Fujitsu', 'Medion']
            first_word = str(titulo).split()[0]
            for brand in brands:
                if first_word.lower() == brand.lower():
                    return brand
            return first_word
        
        df['_brand'] = df['Título'].apply(extract_brand)
        
        _cached_data['computers'] = df
    
    return _cached_data['computers']


def get_featured_data():
    """Load the processed features data if available."""
    try:
        feat_path = DATA_DIR / 'db_features.parquet'
        if feat_path.exists():
            return pd.read_parquet(feat_path)
        # Fallback to CSV
        csv_path = DATA_DIR / 'db_features.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
    except Exception:
        pass
    return get_computers_data()


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Computer Price Prediction API is running'
    })


# ============================================================================
# DESCRIPTIVE ANALYTICS ENDPOINTS
# ============================================================================

@app.route('/api/descriptive/statistics', methods=['GET'])
def get_statistics():
    """Get statistical overview of the marketplace data."""
    df = get_computers_data()
    
    # Price statistics
    price_stats = df['_precio_num'].describe().to_dict()
    
    # Clean up NaN values for JSON
    for key in price_stats:
        if pd.isna(price_stats[key]):
            price_stats[key] = None
    
    # Product type distribution
    product_types = df['Tipo de producto'].value_counts().head(10).to_dict() if 'Tipo de producto' in df.columns else {}
    
    # Brand distribution
    brand_dist = df['_brand'].value_counts().head(15).to_dict()
    
    # Price by product type
    price_by_type = {}
    if 'Tipo de producto' in df.columns:
        for tipo in df['Tipo de producto'].dropna().unique()[:10]:
            subset = df[df['Tipo de producto'] == tipo]['_precio_num'].dropna()
            if len(subset) > 5:
                price_by_type[tipo] = {
                    'mean': float(subset.mean()),
                    'median': float(subset.median()),
                    'min': float(subset.min()),
                    'max': float(subset.max()),
                    'count': int(len(subset))
                }
    
    # Price by brand
    price_by_brand = {}
    for brand in df['_brand'].dropna().unique()[:15]:
        subset = df[df['_brand'] == brand]['_precio_num'].dropna()
        if len(subset) > 5:
            price_by_brand[brand] = {
                'mean': float(subset.mean()),
                'median': float(subset.median()),
                'min': float(subset.min()),
                'max': float(subset.max()),
                'count': int(len(subset))
            }
    
    # RAM distribution
    ram_dist = df['_ram_gb'].value_counts().sort_index().to_dict()
    ram_dist = {str(int(k)): int(v) for k, v in ram_dist.items() if not pd.isna(k)}
    
    # SSD distribution
    ssd_bins = [0, 256, 512, 1000, 2000, 5000]
    ssd_labels = ['0-256GB', '256-512GB', '512GB-1TB', '1-2TB', '2TB+']
    df['_ssd_bin'] = pd.cut(df['_ssd_gb'], bins=ssd_bins, labels=ssd_labels)
    ssd_dist = df['_ssd_bin'].value_counts().to_dict()
    ssd_dist = {str(k): int(v) for k, v in ssd_dist.items() if not pd.isna(k)}
    
    # Screen size distribution
    screen_dist = {}
    for size in [13, 14, 15, 15.6, 16, 17]:
        count = ((df['_screen_inches'] >= size - 0.3) & (df['_screen_inches'] <= size + 0.3)).sum()
        if count > 0:
            screen_dist[str(size)] = int(count)
    
    # Price histogram data
    price_hist = df['_precio_num'].dropna()
    price_bins = np.histogram(price_hist, bins=20)
    price_histogram = {
        'counts': price_bins[0].tolist(),
        'bins': price_bins[1].tolist()
    }
    
    return jsonify({
        'total_listings': int(len(df)),
        'price_statistics': price_stats,
        'product_type_distribution': product_types,
        'brand_distribution': brand_dist,
        'price_by_type': price_by_type,
        'price_by_brand': price_by_brand,
        'ram_distribution': ram_dist,
        'ssd_distribution': ssd_dist,
        'screen_size_distribution': screen_dist,
        'price_histogram': price_histogram,
    })


@app.route('/api/descriptive/clustering', methods=['GET'])
def get_clustering():
    """Get clustering/segmentation data for visualization."""
    df = get_computers_data()
    
    # Create feature matrix for clustering
    cluster_features = ['_precio_num', '_ram_gb', '_ssd_gb', '_screen_inches']
    df_cluster = df[cluster_features].dropna()
    
    if len(df_cluster) < 50:
        return jsonify({'error': 'Not enough data for clustering'}), 400
    
    # Simple K-means clustering
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    # Determine optimal k using elbow method (simplified)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_cluster = df_cluster.copy()
    df_cluster['cluster'] = clusters
    
    # Add product type and brand back for visualization
    df_cluster['_brand'] = df.loc[df_cluster.index, '_brand'].values
    df_cluster['product_type'] = df.loc[df_cluster.index, 'Tipo de producto'].values if 'Tipo de producto' in df.columns else 'Unknown'
    
    # Cluster statistics
    cluster_stats = []
    for i in range(n_clusters):
        subset = df_cluster[df_cluster['cluster'] == i]
        stats = {
            'cluster_id': i,
            'count': int(len(subset)),
            'avg_price': float(subset['_precio_num'].mean()),
            'avg_ram': float(subset['_ram_gb'].mean()),
            'avg_ssd': float(subset['_ssd_gb'].mean()),
            'avg_screen': float(subset['_screen_inches'].mean()),
            'top_brands': subset['_brand'].value_counts().head(3).to_dict(),
            'top_types': subset['product_type'].value_counts().head(3).to_dict() if 'product_type' in subset.columns else {},
        }
        
        # Assign meaningful labels
        if stats['avg_price'] > 2000:
            stats['label'] = 'High-End'
        elif stats['avg_price'] > 1200:
            stats['label'] = 'Premium'
        elif stats['avg_price'] > 800:
            stats['label'] = 'Mid-Range'
        elif stats['avg_price'] > 500:
            stats['label'] = 'Budget'
        else:
            stats['label'] = 'Entry-Level'
        
        cluster_stats.append(stats)
    
    # Sort by avg_price
    cluster_stats.sort(key=lambda x: x['avg_price'], reverse=True)
    
    # Sample data points for visualization (limit to 500 for performance)
    sample_size = min(500, len(df_cluster))
    sample = df_cluster.sample(sample_size, random_state=42)
    
    scatter_data = []
    for _, row in sample.iterrows():
        scatter_data.append({
            'price': float(row['_precio_num']),
            'ram': float(row['_ram_gb']),
            'ssd': float(row['_ssd_gb']),
            'screen': float(row['_screen_inches']),
            'cluster': int(row['cluster']),
            'brand': str(row['_brand']),
        })
    
    return jsonify({
        'n_clusters': n_clusters,
        'cluster_stats': cluster_stats,
        'scatter_data': scatter_data,
    })


# ============================================================================
# PREDICTIVE ANALYTICS ENDPOINTS
# ============================================================================

@app.route('/api/predictive/options', methods=['GET'])
def get_form_options():
    """Get all available options for the prediction form."""
    options = obtener_campos_disponibles()
    
    return jsonify({
        'use_cases': options['use_cases'],
        'brands': options['brands'],
        'cpu_families': options['cpu_families'],
        'gpu_options': options['gpu_options'],
        'resolutions': options['resolutions'],
        'refresh_rates': options['refresh_rates'],
        'ram_options': options['ram_options'],
        'storage_options': options['storage_options'],
        'numeric_ranges': options['numeric_ranges'],
    })


@app.route('/api/predictive/predict', methods=['POST'])
def predict_price():
    """Predict the price for given computer specifications."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Extract use case
    use_case = data.pop('use_case', 'general')
    
    # Clean up the input data
    inputs = {}
    for key, value in data.items():
        if value is not None and value != '':
            inputs[key] = value
    
    try:
        # Get prediction with explanation
        result = explicar_prediccion(inputs, use_case=use_case)
        
        return jsonify({
            'predicted_price': float(result['prediccion']),
            'price_range': {
                'min': float(result['prediccion_min']),
                'max': float(result['prediccion_max']),
            },
            'confidence': result['confidence'],
            'top_features': result['top_features'][:7],  # Top 7 features
            'explanation': result['explanation_text'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictive/model-info', methods=['GET'])
def get_model_metadata():
    """Get information about the loaded model."""
    try:
        info = get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictive/use-case-profiles', methods=['GET'])
def get_use_case_profiles():
    """Get use case profile details."""
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
    return jsonify(profiles)


# ============================================================================
# PRESCRIPTIVE ANALYTICS ENDPOINTS
# ============================================================================

@app.route('/api/prescriptive/similar', methods=['POST'])
def find_similar_offers():
    """Find k-best similar offers to the requested features."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    k = data.pop('k', 10)  # Number of similar offers to return
    
    df = get_computers_data()
    
    # Get predicted price for the input
    use_case = data.pop('use_case', 'general')
    
    try:
        result = explicar_prediccion(data, use_case=use_case)
        predicted_price = result['prediccion']
    except Exception:
        predicted_price = None
    
    # Extract target features
    target_ram = data.get('_ram_gb', 16)
    target_ssd = data.get('_ssd_gb', 512)
    target_screen = data.get('_tamano_pantalla_pulgadas', 15.6)
    target_brand = data.get('_brand', None)
    target_gpu_integrated = data.get('gpu_is_integrated', True)
    
    # Calculate similarity scores
    df_valid = df[
        (df['_precio_num'].notna()) &
        (df['_ram_gb'].notna()) &
        (df['_ssd_gb'].notna())
    ].copy()
    
    if len(df_valid) == 0:
        return jsonify({'error': 'No valid data for comparison'}), 400
    
    # Normalize features for distance calculation
    from sklearn.preprocessing import MinMaxScaler
    
    features = ['_precio_num', '_ram_gb', '_ssd_gb', '_screen_inches']
    
    # Fill missing screen with median
    df_valid['_screen_inches'] = df_valid['_screen_inches'].fillna(df_valid['_screen_inches'].median())
    
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(df_valid[features])
    
    # Target vector
    target = np.array([[
        predicted_price if predicted_price else df_valid['_precio_num'].median(),
        target_ram,
        target_ssd,
        target_screen
    ]])
    target_norm = scaler.transform(target)
    
    # Calculate Euclidean distances
    distances = np.sqrt(((X_norm - target_norm) ** 2).sum(axis=1))
    df_valid['_distance'] = distances
    
    # Apply brand filter if specified
    if target_brand and target_brand != 'Any':
        brand_mask = df_valid['_brand'].str.lower() == target_brand.lower()
        if brand_mask.sum() >= k:
            df_valid = df_valid[brand_mask]
    
    # Get k nearest neighbors
    similar = df_valid.nsmallest(k, '_distance')
    
    # Prepare response
    offers = []
    for idx, row in similar.iterrows():
        offer = {
            'title': str(row['Título']) if pd.notna(row.get('Título')) else 'Unknown',
            'real_price': float(row['_precio_num']),
            'price_range': str(row['Precio_Rango']) if pd.notna(row.get('Precio_Rango')) else None,
            'brand': str(row['_brand']),
            'ram_gb': float(row['_ram_gb']) if pd.notna(row['_ram_gb']) else None,
            'ssd_gb': float(row['_ssd_gb']) if pd.notna(row['_ssd_gb']) else None,
            'screen_inches': float(row['_screen_inches']) if pd.notna(row['_screen_inches']) else None,
            'product_type': str(row['Tipo de producto']) if pd.notna(row.get('Tipo de producto')) else None,
            'series': str(row['Serie']) if pd.notna(row.get('Serie')) else None,
            'distance': float(row['_distance']),
            'price_difference': float(row['_precio_num'] - predicted_price) if predicted_price else None,
        }
        offers.append(offer)
    
    return jsonify({
        'predicted_price': float(predicted_price) if predicted_price else None,
        'target_features': {
            'ram_gb': target_ram,
            'ssd_gb': target_ssd,
            'screen_inches': target_screen,
            'brand': target_brand,
        },
        'similar_offers': offers,
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Starting Computer Price Prediction API...")
    print("API will be available at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)




