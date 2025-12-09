"""
analytics.py - Descriptive and Prescriptive Analytics Functions

This module provides functions for:
- Statistical overview of the laptop market
- Clustering/segmentation of products
- Similarity search for k-best offers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib

# Data paths
DATA_DIR = Path(__file__).parent.parent / 'data'
MODELS_DIR = Path(__file__).parent.parent / 'models'

# Cache for loaded data
_cached_data = None
_cached_features = None


def _to_serializable_number(value):
    """Convert numpy/pandas numeric types to native Python numbers for JSON."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def load_computers_data() -> pd.DataFrame:
    """Load and cache the computers dataset."""
    global _cached_data

    if _cached_data is None:
        csv_path = DATA_DIR / 'db_computers_2025_raw.csv'
        _cached_data = pd.read_csv(csv_path, encoding='utf-8-sig', index_col=0, low_memory=False)
        _cached_data = _cached_data.reset_index(drop=True)

        # Clean price column
        if 'Precio_Rango' in _cached_data.columns:
            _cached_data['_precio_min'] = _cached_data['Precio_Rango'].apply(_extract_min_price)
            _cached_data['_precio_max'] = _cached_data['Precio_Rango'].apply(_extract_max_price)
            _cached_data['_precio_medio'] = (_cached_data['_precio_min'] + _cached_data['_precio_max']) / 2

        # Extract key features
        _cached_data = _extract_features(_cached_data)

    return _cached_data


def _extract_min_price(price_str: str) -> float:
    """Extract minimum price from price range string."""
    if pd.isna(price_str):
        return np.nan
    try:
        # Format: "1.026,53 € – 2.287,17 €"
        price_str = str(price_str).replace('.', '').replace(',', '.').replace('€', '').strip()
        parts = price_str.split('–')
        if parts:
            return float(parts[0].strip())
    except:
        pass
    return np.nan


def _extract_max_price(price_str: str) -> float:
    """Extract maximum price from price range string."""
    if pd.isna(price_str):
        return np.nan
    try:
        price_str = str(price_str).replace('.', '').replace(',', '.').replace('€', '').strip()
        parts = price_str.split('–')
        if len(parts) > 1:
            return float(parts[1].strip())
        elif parts:
            return float(parts[0].strip())
    except:
        pass
    return np.nan


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean key features from raw data."""
    df = df.copy()

    # Extract RAM
    if 'RAM_Memoria RAM' in df.columns:
        df['_ram_gb'] = df['RAM_Memoria RAM'].apply(_parse_memory)

    # Extract Storage
    if 'Disco duro_Capacidad de memoria SSD' in df.columns:
        df['_ssd_gb'] = df['Disco duro_Capacidad de memoria SSD'].apply(_parse_storage)

    # Extract Screen Size
    if 'Pantalla_Diagonal de la pantalla' in df.columns:
        df['_screen_size'] = df['Pantalla_Diagonal de la pantalla'].apply(_parse_screen_size)

    # Extract Weight
    if 'Medidas y peso_Peso' in df.columns:
        df['_weight_kg'] = df['Medidas y peso_Peso'].apply(_parse_weight)

    # Extract CPU Brand
    if 'Procesador_Fabricante del procesador' in df.columns:
        df['_cpu_brand'] = df['Procesador_Fabricante del procesador'].fillna('Unknown')

    # Extract GPU Brand
    if 'Gráfica_Fabricante de la tarjeta gráfica' in df.columns:
        df['_gpu_brand'] = df['Gráfica_Fabricante de la tarjeta gráfica'].fillna('Integrated')

    # Extract Brand from Serie
    if 'Serie' in df.columns:
        df['_brand'] = df['Serie'].apply(_extract_brand)

    # Extract Refresh Rate
    if 'Pantalla_Tasa de actualización de imagen' in df.columns:
        df['_refresh_rate'] = df['Pantalla_Tasa de actualización de imagen'].apply(_parse_refresh_rate)

    return df


def _clean_string(s: str) -> str:
    """Clean string by removing non-breaking spaces and normalizing."""
    import re
    if pd.isna(s):
        return ''
    # Replace non-breaking spaces and other unicode spaces
    s = str(s).replace('\xa0', ' ').replace('\u00a0', ' ')
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _parse_memory(mem_str) -> float:
    """Parse memory string to GB."""
    if pd.isna(mem_str):
        return np.nan
    try:
        mem_str = _clean_string(mem_str).lower()
        # Remove "ram" suffix
        mem_str = mem_str.replace('ram', '').strip()
        if 'gb' in mem_str:
            return float(mem_str.replace('gb', '').strip())
        elif 'tb' in mem_str:
            return float(mem_str.replace('tb', '').strip()) * 1024
    except:
        pass
    return np.nan


def _parse_storage(storage_str) -> float:
    """Parse storage string to GB."""
    if pd.isna(storage_str):
        return np.nan
    try:
        storage_str = _clean_string(storage_str).lower().replace(',', '.')
        if 'tb' in storage_str:
            val = float(storage_str.replace('tb', '').strip())
            return val * 1000
        elif 'gb' in storage_str:
            return float(storage_str.replace('gb', '').strip())
    except:
        pass
    return np.nan


def _parse_screen_size(size_str) -> float:
    """Parse screen size to inches."""
    if pd.isna(size_str):
        return np.nan
    try:
        size_str = _clean_string(size_str).lower().replace(',', '.')
        if 'cm' in size_str:
            cm = float(size_str.replace('cm', '').strip())
            return cm / 2.54  # Convert to inches
        elif 'pulgadas' in size_str or '"' in size_str:
            return float(size_str.replace('pulgadas', '').replace('"', '').strip())
    except:
        pass
    return np.nan


def _parse_weight(weight_str) -> float:
    """Parse weight to kg."""
    if pd.isna(weight_str):
        return np.nan
    try:
        weight_str = _clean_string(weight_str).lower().replace(',', '.')
        if 'kg' in weight_str:
            return float(weight_str.replace('kg', '').strip())
        elif 'g' in weight_str:
            return float(weight_str.replace('g', '').strip()) / 1000
    except:
        pass
    return np.nan


def _parse_refresh_rate(rate_str) -> float:
    """Parse refresh rate to Hz."""
    if pd.isna(rate_str):
        return np.nan
    try:
        rate_str = str(rate_str).lower()
        if 'hz' in rate_str:
            return float(rate_str.replace('hz', '').strip())
    except:
        pass
    return np.nan


def _extract_brand(serie_str) -> str:
    """Extract brand from Serie column."""
    if pd.isna(serie_str):
        return 'Unknown'

    serie_str = str(serie_str).lower()

    brands = {
        'apple': 'Apple',
        'asus': 'ASUS',
        'acer': 'Acer',
        'dell': 'Dell',
        'hp': 'HP',
        'lenovo': 'Lenovo',
        'msi': 'MSI',
        'razer': 'Razer',
        'samsung': 'Samsung',
        'microsoft': 'Microsoft',
        'lg': 'LG',
        'huawei': 'Huawei',
        'toshiba': 'Toshiba',
        'gigabyte': 'Gigabyte',
        'alienware': 'Alienware',
        'medion': 'Medion',
    }

    for key, value in brands.items():
        if key in serie_str:
            return value

    return 'Other'


# =============================================================================
# STATISTICAL OVERVIEW
# =============================================================================

def get_market_statistics() -> Dict[str, Any]:
    """Get statistical overview of the laptop market."""
    df = load_computers_data()

    stats = {
        'total_listings': len(df),
        'price_stats': {},
        'brand_distribution': {},
        'cpu_distribution': {},
        'gpu_distribution': {},
        'ram_distribution': {},
        'storage_distribution': {},
        'screen_distribution': {},
    }

    # Price statistics
    if '_precio_medio' in df.columns:
        valid_prices = df['_precio_medio'].dropna()
        stats['price_stats'] = {
            'mean': valid_prices.mean(),
            'median': valid_prices.median(),
            'min': valid_prices.min(),
            'max': valid_prices.max(),
            'std': valid_prices.std(),
            'q25': valid_prices.quantile(0.25),
            'q75': valid_prices.quantile(0.75),
        }

    # Brand distribution
    if '_brand' in df.columns:
        brand_counts = df['_brand'].value_counts().head(10)
        stats['brand_distribution'] = brand_counts.to_dict()

    # CPU distribution
    if '_cpu_brand' in df.columns:
        cpu_counts = df['_cpu_brand'].value_counts().head(10)
        stats['cpu_distribution'] = cpu_counts.to_dict()

    # GPU distribution
    if '_gpu_brand' in df.columns:
        gpu_counts = df['_gpu_brand'].value_counts().head(10)
        stats['gpu_distribution'] = gpu_counts.to_dict()

    # RAM distribution
    if '_ram_gb' in df.columns:
        ram_bins = pd.cut(df['_ram_gb'].dropna(), bins=[0, 8, 16, 32, 64, 128, float('inf')],
                          labels=['≤8GB', '9-16GB', '17-32GB', '33-64GB', '65-128GB', '>128GB'])
        stats['ram_distribution'] = ram_bins.value_counts().to_dict()

    # Storage distribution
    if '_ssd_gb' in df.columns:
        storage_bins = pd.cut(df['_ssd_gb'].dropna(), bins=[0, 256, 512, 1000, 2000, float('inf')],
                              labels=['≤256GB', '257-512GB', '513GB-1TB', '1-2TB', '>2TB'])
        stats['storage_distribution'] = storage_bins.value_counts().to_dict()

    # Screen size distribution
    if '_screen_size' in df.columns:
        screen_bins = pd.cut(df['_screen_size'].dropna(), bins=[0, 13, 14, 15, 16, 17, float('inf')],
                             labels=['≤13"', '13-14"', '14-15"', '15-16"', '16-17"', '>17"'])
        stats['screen_distribution'] = screen_bins.value_counts().to_dict()

    return stats


def get_descriptive_statistics(histogram_bins: int = 20) -> Dict[str, Any]:
    """Return detailed descriptive analytics needed by the React dashboard."""
    df = load_computers_data()

    if '_precio_medio' not in df.columns:
        raise ValueError("Price data not available")

    price_series = df['_precio_medio'].dropna()
    if price_series.empty:
        raise ValueError("Price data not available")

    price_stats = price_series.describe().to_dict()
    price_stats = {k: _to_serializable_number(v) for k, v in price_stats.items()}

    product_types = {}
    if 'Tipo de producto' in df.columns:
        product_types = (
            df['Tipo de producto']
            .dropna()
            .value_counts()
            .head(10)
            .astype(int)
            .to_dict()
        )

    brand_distribution = {}
    if '_brand' in df.columns:
        brand_distribution = (
            df['_brand']
            .dropna()
            .value_counts()
            .head(15)
            .astype(int)
            .to_dict()
        )

    price_by_type = {}
    if 'Tipo de producto' in df.columns:
        type_prices = df[['Tipo de producto', '_precio_medio']].dropna()
        for tipo, subset in type_prices.groupby('Tipo de producto'):
            if len(subset) < 5:
                continue
            stats = subset['_precio_medio'].describe()[['mean', '50%', 'min', 'max', 'count']]
            price_by_type[str(tipo)] = {
                'mean': _to_serializable_number(stats['mean']),
                'median': _to_serializable_number(stats['50%']),
                'min': _to_serializable_number(stats['min']),
                'max': _to_serializable_number(stats['max']),
                'count': int(stats['count']),
            }

    price_by_brand = {}
    if '_brand' in df.columns:
        brand_prices = df[['_brand', '_precio_medio']].dropna()
        for brand, subset in brand_prices.groupby('_brand'):
            if len(subset) < 5:
                continue
            stats = subset['_precio_medio'].describe()[['mean', '50%', 'min', 'max', 'count']]
            price_by_brand[str(brand)] = {
                'mean': _to_serializable_number(stats['mean']),
                'median': _to_serializable_number(stats['50%']),
                'min': _to_serializable_number(stats['min']),
                'max': _to_serializable_number(stats['max']),
                'count': int(stats['count']),
            }

    ram_distribution = {}
    if '_ram_gb' in df.columns:
        ram_counts = (
            df['_ram_gb']
            .dropna()
            .round()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        ram_distribution = {str(k): int(v) for k, v in ram_counts.items()}

    ssd_distribution = {}
    if '_ssd_gb' in df.columns:
        ssd_bins = [0, 256, 512, 1000, 2000, 5000]
        ssd_labels = ['0-256GB', '256-512GB', '512GB-1TB', '1-2TB', '2TB+']
        ssd_series = pd.cut(df['_ssd_gb'], bins=ssd_bins, labels=ssd_labels)
        counts = ssd_series.value_counts()
        ssd_distribution = {str(k): int(v) for k, v in counts.items() if not pd.isna(k)}

    screen_distribution = {}
    if '_screen_size' in df.columns:
        sizes = df['_screen_size'].dropna()
        for size in [13, 14, 15, 15.6, 16, 17]:
            count = ((sizes >= size - 0.3) & (sizes <= size + 0.3)).sum()
            if count:
                screen_distribution[str(size)] = int(count)

    price_hist = np.histogram(price_series, bins=histogram_bins)
    price_histogram = {
        'counts': [int(x) for x in price_hist[0]],
        'bins': [float(x) for x in price_hist[1]],
    }

    return {
        'total_listings': int(len(df)),
        'price_statistics': price_stats,
        'product_type_distribution': product_types,
        'brand_distribution': brand_distribution,
        'price_by_type': price_by_type,
        'price_by_brand': price_by_brand,
        'ram_distribution': ram_distribution,
        'ssd_distribution': ssd_distribution,
        'screen_size_distribution': screen_distribution,
        'price_histogram': price_histogram,
    }


def get_price_by_category(category: str) -> pd.DataFrame:
    """Get price statistics grouped by a category."""
    df = load_computers_data()

    category_map = {
        'brand': '_brand',
        'cpu': '_cpu_brand',
        'gpu': '_gpu_brand',
    }

    col = category_map.get(category, category)

    if col not in df.columns or '_precio_medio' not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(col)['_precio_medio'].agg(['mean', 'median', 'min', 'max', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    grouped.columns = ['Mean Price', 'Median Price', 'Min Price', 'Max Price', 'Count']

    return grouped.reset_index()


# =============================================================================
# CLUSTERING / SEGMENTATION
# =============================================================================

def get_clustering_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare data and perform clustering."""
    df = load_computers_data()

    # Select features for clustering
    feature_cols = ['_precio_medio', '_ram_gb', '_ssd_gb', '_screen_size', '_weight_kg']
    available_cols = [c for c in feature_cols if c in df.columns]

    if len(available_cols) < 3:
        return df, np.array([]), np.array([])

    # Filter to rows with complete data
    cluster_df = df[available_cols].dropna()

    if len(cluster_df) < 100:
        return df, np.array([]), np.array([])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    # Perform K-Means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Add cluster labels back to dataframe
    df.loc[cluster_df.index, '_cluster'] = clusters

    return df, X_pca, clusters


def get_cluster_profiles() -> List[Dict[str, Any]]:
    """Get profile/description for each cluster."""
    df, X_pca, clusters = get_clustering_data()

    if '_cluster' not in df.columns:
        return []

    profiles = []

    for cluster_id in sorted(df['_cluster'].dropna().unique()):
        cluster_data = df[df['_cluster'] == cluster_id]

        profile = {
            'cluster_id': int(cluster_id),
            'count': len(cluster_data),
            'percentage': len(cluster_data) / len(df[df['_cluster'].notna()]) * 100,
            'avg_price': cluster_data['_precio_medio'].mean() if '_precio_medio' in cluster_data else 0,
            'avg_ram': cluster_data['_ram_gb'].mean() if '_ram_gb' in cluster_data else 0,
            'avg_storage': cluster_data['_ssd_gb'].mean() if '_ssd_gb' in cluster_data else 0,
            'avg_screen': cluster_data['_screen_size'].mean() if '_screen_size' in cluster_data else 0,
            'top_brand': cluster_data['_brand'].mode().iloc[0] if '_brand' in cluster_data and len(cluster_data['_brand'].mode()) > 0 else 'N/A',
        }

        # Determine segment name based on characteristics
        if profile['avg_price'] > 2000:
            profile['segment_name'] = 'Premium/High-End'
            profile['description'] = 'High-performance laptops with premium specs and pricing'
        elif profile['avg_price'] > 1200:
            profile['segment_name'] = 'Performance'
            profile['description'] = 'Strong all-around performers for demanding users'
        elif profile['avg_price'] > 800:
            profile['segment_name'] = 'Mainstream'
            profile['description'] = 'Balanced laptops for everyday productivity'
        elif profile['avg_price'] > 500:
            profile['segment_name'] = 'Budget'
            profile['description'] = 'Affordable options for basic computing needs'
        else:
            profile['segment_name'] = 'Entry-Level'
            profile['description'] = 'Basic laptops for light use'

        profiles.append(profile)

    return sorted(profiles, key=lambda x: x['avg_price'], reverse=True)


def get_clustering_summary(max_points: int = 500) -> Dict[str, Any]:
    """Return clustering data formatted for the dashboard."""
    df, _, clusters = get_clustering_data()

    if clusters is None or len(clusters) == 0:
        raise ValueError("Not enough data for clustering")

    clustered_df = df[df['_cluster'].notna()].copy()
    if clustered_df.empty:
        raise ValueError("Not enough data for clustering")

    cluster_stats = []
    for cluster_id in sorted(clustered_df['_cluster'].dropna().unique()):
        subset = clustered_df[clustered_df['_cluster'] == cluster_id]
        if subset.empty:
            continue

        avg_price = subset['_precio_medio'].dropna().mean() if '_precio_medio' in subset else 0
        avg_ram = subset['_ram_gb'].dropna().median() if '_ram_gb' in subset else 0
        avg_ssd = subset['_ssd_gb'].dropna().median() if '_ssd_gb' in subset else 0
        avg_screen = subset['_screen_size'].dropna().mean() if '_screen_size' in subset else 0

        label = 'Entry-Level'
        if avg_price > 2000:
            label = 'High-End'
        elif avg_price > 1200:
            label = 'Premium'
        elif avg_price > 800:
            label = 'Mid-Range'
        elif avg_price > 500:
            label = 'Budget'

        cluster_stats.append({
            'cluster_id': int(cluster_id),
            'count': int(len(subset)),
            'avg_price': float(avg_price) if not np.isnan(avg_price) else 0.0,
            'avg_ram': float(avg_ram) if not np.isnan(avg_ram) else 0.0,
            'avg_ssd': float(avg_ssd) if not np.isnan(avg_ssd) else 0.0,
            'avg_screen': float(avg_screen) if not np.isnan(avg_screen) else 0.0,
            'top_brands': (
                {str(k): int(v) for k, v in subset['_brand'].value_counts().head(3).to_dict().items()}
                if '_brand' in subset else {}
            ),
            'label': label,
        })

    cluster_stats.sort(key=lambda x: x['avg_price'], reverse=True)

    sample_size = min(max_points, len(clustered_df))
    sampled_df = clustered_df.sample(sample_size, random_state=42) if sample_size > 0 else clustered_df

    scatter_data = []
    for _, row in sampled_df.iterrows():
        if pd.isna(row.get('_ram_gb')) or pd.isna(row.get('_ssd_gb')) or pd.isna(row.get('_precio_medio')):
            continue
        scatter_data.append({
            'price': float(row['_precio_medio']),
            'ram': float(row['_ram_gb']),
            'ssd': float(row['_ssd_gb']),
            'screen': float(row['_screen_size']) if pd.notna(row.get('_screen_size')) else None,
            'cluster': int(row.get('_cluster', 0)),
            'brand': str(row.get('_brand', 'Unknown')),
        })

    return {
        'n_clusters': len(cluster_stats),
        'cluster_stats': cluster_stats,
        'scatter_data': scatter_data,
    }


# =============================================================================
# SIMILARITY SEARCH (PRESCRIPTIVE)
# =============================================================================

def find_similar_laptops(specs: Dict[str, Any], k: int = 5) -> pd.DataFrame:
    """
    Find k most similar laptops to the given specifications.

    Parameters
    ----------
    specs : Dict[str, Any]
        Specification dictionary with keys like:
        - price_target: Target price
        - ram_gb: RAM in GB
        - ssd_gb: Storage in GB
        - screen_size: Screen size in inches
        - brand: Preferred brand (optional)
        - cpu_brand: CPU brand preference (optional)
        - gpu_brand: GPU brand preference (optional)
    k : int
        Number of similar laptops to return

    Returns
    -------
    pd.DataFrame
        DataFrame with k most similar laptops and their distances
    """
    df = load_computers_data()

    # Features for similarity matching
    feature_cols = ['_precio_medio', '_ram_gb', '_ssd_gb', '_screen_size']
    available_cols = [c for c in feature_cols if c in df.columns]

    if len(available_cols) < 2:
        return pd.DataFrame()

    # Filter to complete data
    valid_df = df.dropna(subset=available_cols).copy()

    if len(valid_df) < k:
        return pd.DataFrame()

    # Apply filters if specified
    if 'brand' in specs and specs['brand']:
        brand_filter = valid_df['_brand'].str.lower() == str(specs['brand']).lower()
        if brand_filter.sum() >= k:
            valid_df = valid_df[brand_filter]

    if 'cpu_brand' in specs and specs['cpu_brand']:
        cpu_filter = valid_df['_cpu_brand'].str.lower().str.contains(str(specs['cpu_brand']).lower(), na=False)
        if cpu_filter.sum() >= k:
            valid_df = valid_df[cpu_filter]

    # Build feature matrix
    X = valid_df[available_cols].values

    # Normalize for fair distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build query point
    query = []
    for col in available_cols:
        if col == '_precio_medio' and 'price_target' in specs:
            query.append(specs['price_target'])
        elif col == '_ram_gb' and 'ram_gb' in specs:
            query.append(specs['ram_gb'])
        elif col == '_ssd_gb' and 'ssd_gb' in specs:
            query.append(specs['ssd_gb'])
        elif col == '_screen_size' and 'screen_size' in specs:
            query.append(specs['screen_size'])
        else:
            # Use median as default
            query.append(valid_df[col].median())

    query_scaled = scaler.transform([query])

    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(valid_df)), metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(query_scaled)

    # Build results
    results = valid_df.iloc[indices[0]].copy()
    results['_similarity_distance'] = distances[0]
    results['_similarity_rank'] = range(1, len(results) + 1)

    # Calculate feature-wise differences
    for i, col in enumerate(available_cols):
        col_name = col.replace('_', '')
        results[f'_diff_{col_name}'] = results[col] - query[i]

    # Select output columns
    output_cols = [
        'Título', '_precio_medio', '_precio_min', '_precio_max',
        '_brand', '_ram_gb', '_ssd_gb', '_screen_size', '_weight_kg',
        '_cpu_brand', '_gpu_brand', 'Tipo de producto',
        '_similarity_distance', '_similarity_rank'
    ]
    output_cols = [c for c in output_cols if c in results.columns]

    return results[output_cols].reset_index(drop=True)


def get_sample_laptops(n: int = 10, segment: str = None) -> pd.DataFrame:
    """Get sample laptops, optionally filtered by segment."""
    df = load_computers_data()

    # Ensure we have cluster data
    if '_cluster' not in df.columns:
        df, _, _ = get_clustering_data()

    if segment and '_cluster' in df.columns:
        # Filter by cluster/segment
        profiles = get_cluster_profiles()
        cluster_id = None
        for p in profiles:
            if segment.lower() in p['segment_name'].lower():
                cluster_id = p['cluster_id']
                break

        if cluster_id is not None:
            df = df[df['_cluster'] == cluster_id]

    # Select columns
    output_cols = [
        'Título', '_precio_medio', '_brand', '_ram_gb', '_ssd_gb',
        '_screen_size', '_cpu_brand', '_gpu_brand'
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    return df[output_cols].dropna().sample(min(n, len(df))).reset_index(drop=True)
