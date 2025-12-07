"""
benchmark_cache.py - Cached Benchmark Data Loading

This module provides cached loading of CPU and GPU benchmark data
for fast lookups in the frontend application.
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional

# Data directory path
DATA_DIR = Path(__file__).parent.parent / 'data'


@lru_cache(maxsize=1)
def get_cpu_data() -> pd.DataFrame:
    """
    Load and cache CPU benchmark data.

    Returns
    -------
    pd.DataFrame
        CPU benchmark data with columns: CPU Name, cpuMark, Rank, cpuValue, Price
    """
    cpu_path = DATA_DIR / 'db_cpu_raw.csv'
    if not cpu_path.exists():
        raise FileNotFoundError(f"CPU benchmark file not found: {cpu_path}")

    return pd.read_csv(cpu_path, encoding='utf-8-sig', index_col=0).reset_index(drop=True)


@lru_cache(maxsize=1)
def get_gpu_data() -> pd.DataFrame:
    """
    Load and cache GPU benchmark data.

    Returns
    -------
    pd.DataFrame
        GPU benchmark data with columns: Videocard Name, G3D Mark, Rank, Value, Price
    """
    gpu_path = DATA_DIR / 'db_gpu_raw.csv'
    if not gpu_path.exists():
        raise FileNotFoundError(f"GPU benchmark file not found: {gpu_path}")

    return pd.read_csv(gpu_path, encoding='utf-8-sig', index_col=0).reset_index(drop=True)


@lru_cache(maxsize=1)
def get_cpu_names() -> List[str]:
    """
    Get list of all CPU names for autocomplete.

    Returns
    -------
    List[str]
        Sorted list of CPU names
    """
    df = get_cpu_data()
    return sorted(df['CPU Name'].dropna().unique().tolist())


@lru_cache(maxsize=1)
def get_gpu_names() -> List[str]:
    """
    Get list of all GPU names for autocomplete.

    Returns
    -------
    List[str]
        Sorted list of GPU names
    """
    df = get_gpu_data()
    return sorted(df['Videocard Name'].dropna().unique().tolist())


def get_brand_options() -> List[str]:
    """
    Get list of laptop brand options.

    Returns
    -------
    List[str]
        List of brand names
    """
    return [
        'Apple', 'ASUS', 'Acer', 'Dell', 'HP', 'Lenovo', 'MSI',
        'Samsung', 'Microsoft', 'Razer', 'LG', 'Huawei', 'Gigabyte',
        'Toshiba', 'Fujitsu', 'Medion', 'Other'
    ]


def get_cpu_brand_options() -> List[str]:
    """Get CPU brand options."""
    return ['Intel', 'AMD', 'Apple', 'Qualcomm']


def get_cpu_family_options(brand: str = None) -> List[str]:
    """
    Get CPU family options, optionally filtered by brand.

    Parameters
    ----------
    brand : str, optional
        Filter by brand (Intel, AMD, Apple)

    Returns
    -------
    List[str]
        List of CPU family names
    """
    families = {
        'Intel': ['Core i3', 'Core i5', 'Core i7', 'Core i9', 'Core Ultra 5',
                  'Core Ultra 7', 'Core Ultra 9', 'Celeron', 'Pentium', 'Atom'],
        'AMD': ['Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9', 'Athlon'],
        'Apple': ['M1', 'M1 Pro', 'M1 Max', 'M2', 'M2 Pro', 'M2 Max',
                  'M3', 'M3 Pro', 'M3 Max', 'M4', 'M4 Pro', 'M4 Max'],
        'Qualcomm': ['Snapdragon X Elite', 'Snapdragon X Plus'],
    }

    if brand and brand in families:
        return families[brand]

    # Return all families
    all_families = []
    for fam_list in families.values():
        all_families.extend(fam_list)
    return sorted(set(all_families))


def get_gpu_brand_options() -> List[str]:
    """Get GPU brand options."""
    return ['Integrated', 'NVIDIA', 'AMD', 'Intel Arc']


def get_gpu_series_options(brand: str = None, integrated: bool = False) -> List[str]:
    """
    Get GPU series options, optionally filtered by brand.

    Parameters
    ----------
    brand : str, optional
        Filter by brand (NVIDIA, AMD, Intel)
    integrated : bool
        If True, return integrated GPU options

    Returns
    -------
    List[str]
        List of GPU series names
    """
    if integrated:
        return ['Intel UHD Graphics', 'Intel Iris Xe', 'AMD Radeon Graphics', 'Apple GPU']

    series = {
        'NVIDIA': [
            'RTX 4090', 'RTX 4080', 'RTX 4070', 'RTX 4060', 'RTX 4050',
            'RTX 3080', 'RTX 3070', 'RTX 3060', 'RTX 3050',
            'GTX 1660 Ti', 'GTX 1650',
            'MX 550', 'MX 450'
        ],
        'AMD': [
            'RX 7900', 'RX 7700', 'RX 7600',
            'RX 6800', 'RX 6700', 'RX 6600', 'RX 6500'
        ],
        'Intel Arc': ['Arc A770', 'Arc A750', 'Arc A580', 'Arc A380'],
    }

    if brand and brand in series:
        return series[brand]

    # Return all series
    all_series = []
    for s_list in series.values():
        all_series.extend(s_list)
    return all_series


def get_resolution_options() -> Dict[str, int]:
    """
    Get screen resolution options with pixel counts.

    Returns
    -------
    Dict[str, int]
        Mapping of resolution name to total pixels
    """
    return {
        'HD (1366x768)': 1049088,
        'HD+ (1600x900)': 1440000,
        'FHD (1920x1080)': 2073600,
        'FHD+ (1920x1200)': 2304000,
        'QHD (2560x1440)': 3686400,
        'QHD+ (2560x1600)': 4096000,
        '4K UHD (3840x2160)': 8294400,
        '5K (5120x2880)': 14745600,
    }


def get_refresh_rate_options() -> List[int]:
    """Get common refresh rate options."""
    return [60, 90, 120, 144, 165, 240, 360]


def get_ram_options() -> List[int]:
    """Get common RAM size options in GB."""
    return [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def get_storage_options() -> List[int]:
    """Get common storage size options in GB."""
    return [128, 256, 512, 1000, 2000, 4000]


def get_screen_size_range() -> tuple:
    """Get screen size range (min, max) in inches."""
    return (11.0, 18.0)


def get_weight_range() -> tuple:
    """Get weight range (min, max) in kg."""
    return (0.5, 4.5)


def clear_cache():
    """Clear all cached data."""
    get_cpu_data.cache_clear()
    get_gpu_data.cache_clear()
    get_cpu_names.cache_clear()
    get_gpu_names.cache_clear()
