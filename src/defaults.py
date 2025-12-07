"""
defaults.py - Use-Case Based Defaults for Missing Value Handling

This module provides sensible default values for laptop specifications
based on the user's intended use case. This enables Simple Mode predictions
where users only need to provide a few key variables.
"""

from typing import Dict, Any, Optional

# Use-case profiles with typical specifications
USE_CASE_PROFILES = {
    'gaming': {
        '_ram_gb': 16,
        '_ssd_gb': 512,
        '_tasa_refresco_hz': 144,
        '_tamano_pantalla_pulgadas': 15.6,
        '_peso_kg': 2.3,
        '_resolucion_pixeles': 2073600,  # FHD
        '_gpu_memory_gb': 6,
        '_cpu_cores': 8,
        'gpu_is_integrated': False,
        'cpu_brand': 'intel',
        'cpu_family': 'core i7',
        'gpu_brand': 'nvidia',
        'gpu_series': 'rtx',
        '_tiene_wifi': 1,
        '_tiene_bluetooth': 1,
        '_tiene_webcam': 1,
    },
    'work': {
        '_ram_gb': 16,
        '_ssd_gb': 256,
        '_tasa_refresco_hz': 60,
        '_tamano_pantalla_pulgadas': 14.0,
        '_peso_kg': 1.5,
        '_resolucion_pixeles': 2073600,  # FHD
        '_gpu_memory_gb': 0,
        '_cpu_cores': 6,
        'gpu_is_integrated': True,
        'cpu_brand': 'intel',
        'cpu_family': 'core i5',
        'gpu_brand': 'intel',
        'gpu_series': 'iris',
        '_tiene_wifi': 1,
        '_tiene_bluetooth': 1,
        '_tiene_webcam': 1,
    },
    'creative': {
        '_ram_gb': 32,
        '_ssd_gb': 1000,
        '_tasa_refresco_hz': 60,
        '_tamano_pantalla_pulgadas': 16.0,
        '_peso_kg': 2.0,
        '_resolucion_pixeles': 3686400,  # QHD
        '_gpu_memory_gb': 8,
        '_cpu_cores': 10,
        'gpu_is_integrated': False,
        'cpu_brand': 'intel',
        'cpu_family': 'core i7',
        'gpu_brand': 'nvidia',
        'gpu_series': 'rtx',
        '_tiene_wifi': 1,
        '_tiene_bluetooth': 1,
        '_tiene_webcam': 1,
    },
    'student': {
        '_ram_gb': 8,
        '_ssd_gb': 256,
        '_tasa_refresco_hz': 60,
        '_tamano_pantalla_pulgadas': 14.0,
        '_peso_kg': 1.4,
        '_resolucion_pixeles': 2073600,  # FHD
        '_gpu_memory_gb': 0,
        '_cpu_cores': 4,
        'gpu_is_integrated': True,
        'cpu_brand': 'intel',
        'cpu_family': 'core i5',
        'gpu_brand': 'intel',
        'gpu_series': 'uhd',
        '_tiene_wifi': 1,
        '_tiene_bluetooth': 1,
        '_tiene_webcam': 1,
    },
    'general': {
        '_ram_gb': 16,
        '_ssd_gb': 512,
        '_tasa_refresco_hz': 60,
        '_tamano_pantalla_pulgadas': 15.6,
        '_peso_kg': 1.8,
        '_resolucion_pixeles': 2073600,  # FHD
        '_gpu_memory_gb': 0,
        '_cpu_cores': 6,
        'gpu_is_integrated': True,
        'cpu_brand': 'intel',
        'cpu_family': 'core i5',
        'gpu_brand': 'intel',
        'gpu_series': 'uhd',
        '_tiene_wifi': 1,
        '_tiene_bluetooth': 1,
        '_tiene_webcam': 1,
    }
}

# Default benchmark values when CPU/GPU lookup fails
# These are approximate median values from the training data
BENCHMARK_DEFAULTS = {
    # CPU defaults by family
    'cpu_benchmarks': {
        'core i3': {'_cpu_mark': 8000, '_cpu_rank': 800, '_cpu_value': 50, '_cpu_price_usd': 150},
        'core i5': {'_cpu_mark': 15000, '_cpu_rank': 400, '_cpu_value': 60, '_cpu_price_usd': 250},
        'core i7': {'_cpu_mark': 22000, '_cpu_rank': 200, '_cpu_value': 55, '_cpu_price_usd': 350},
        'core i9': {'_cpu_mark': 35000, '_cpu_rank': 50, '_cpu_value': 40, '_cpu_price_usd': 500},
        'ryzen 5': {'_cpu_mark': 18000, '_cpu_rank': 300, '_cpu_value': 80, '_cpu_price_usd': 200},
        'ryzen 7': {'_cpu_mark': 25000, '_cpu_rank': 150, '_cpu_value': 70, '_cpu_price_usd': 300},
        'ryzen 9': {'_cpu_mark': 38000, '_cpu_rank': 30, '_cpu_value': 50, '_cpu_price_usd': 450},
        'm1': {'_cpu_mark': 15000, '_cpu_rank': 350, '_cpu_value': 60, '_cpu_price_usd': 0},
        'm2': {'_cpu_mark': 20000, '_cpu_rank': 200, '_cpu_value': 65, '_cpu_price_usd': 0},
        'm3': {'_cpu_mark': 25000, '_cpu_rank': 100, '_cpu_value': 60, '_cpu_price_usd': 0},
        'default': {'_cpu_mark': 15000, '_cpu_rank': 400, '_cpu_value': 55, '_cpu_price_usd': 250},
    },
    # GPU defaults by series (for discrete GPUs)
    'gpu_benchmarks': {
        'rtx 4090': {'_gpu_mark': 39000, '_gpu_rank': 1, '_gpu_value': 20, '_gpu_price_usd': 1600},
        'rtx 4080': {'_gpu_mark': 34000, '_gpu_rank': 5, '_gpu_value': 25, '_gpu_price_usd': 1200},
        'rtx 4070': {'_gpu_mark': 26000, '_gpu_rank': 15, '_gpu_value': 40, '_gpu_price_usd': 600},
        'rtx 4060': {'_gpu_mark': 19000, '_gpu_rank': 40, '_gpu_value': 60, '_gpu_price_usd': 300},
        'rtx 4050': {'_gpu_mark': 14000, '_gpu_rank': 80, '_gpu_value': 70, '_gpu_price_usd': 200},
        'rtx 3080': {'_gpu_mark': 24000, '_gpu_rank': 20, '_gpu_value': 30, '_gpu_price_usd': 700},
        'rtx 3070': {'_gpu_mark': 21000, '_gpu_rank': 30, '_gpu_value': 45, '_gpu_price_usd': 500},
        'rtx 3060': {'_gpu_mark': 17000, '_gpu_rank': 50, '_gpu_value': 65, '_gpu_price_usd': 300},
        'rtx 3050': {'_gpu_mark': 12000, '_gpu_rank': 100, '_gpu_value': 80, '_gpu_price_usd': 200},
        'gtx 1650': {'_gpu_mark': 7500, '_gpu_rank': 150, '_gpu_value': 90, '_gpu_price_usd': 150},
        'rx 7600': {'_gpu_mark': 18000, '_gpu_rank': 45, '_gpu_value': 70, '_gpu_price_usd': 270},
        'rx 6600': {'_gpu_mark': 15000, '_gpu_rank': 70, '_gpu_value': 75, '_gpu_price_usd': 200},
        'integrated': {'_gpu_mark': None, '_gpu_rank': None, '_gpu_value': None, '_gpu_price_usd': None},
        'default': {'_gpu_mark': 12000, '_gpu_rank': 100, '_gpu_value': 70, '_gpu_price_usd': 200},
    }
}

# Brand mapping for display
BRAND_DISPLAY_NAMES = {
    'apple': 'Apple',
    'asus': 'ASUS',
    'acer': 'Acer',
    'dell': 'Dell',
    'hp': 'HP',
    'lenovo': 'Lenovo',
    'msi': 'MSI',
    'samsung': 'Samsung',
    'microsoft': 'Microsoft',
    'razer': 'Razer',
    'lg': 'LG',
    'huawei': 'Huawei',
}

# CPU family options for UI
CPU_FAMILY_OPTIONS = {
    'Intel': ['Core i3', 'Core i5', 'Core i7', 'Core i9', 'Celeron', 'Pentium'],
    'AMD': ['Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9', 'Athlon'],
    'Apple': ['M1', 'M1 Pro', 'M1 Max', 'M2', 'M2 Pro', 'M2 Max', 'M3', 'M3 Pro', 'M3 Max'],
}

# GPU options for UI
GPU_OPTIONS = {
    'Integrated': ['Intel UHD', 'Intel Iris Xe', 'AMD Radeon Graphics', 'Apple GPU'],
    'NVIDIA': ['RTX 4090', 'RTX 4080', 'RTX 4070', 'RTX 4060', 'RTX 4050',
               'RTX 3080', 'RTX 3070', 'RTX 3060', 'RTX 3050',
               'GTX 1650', 'GTX 1660 Ti'],
    'AMD': ['RX 7600', 'RX 6700', 'RX 6600', 'RX 6500'],
}


def fill_missing_with_defaults(user_inputs: Dict[str, Any],
                                use_case: str = 'general') -> Dict[str, Any]:
    """
    Fill missing values with use-case appropriate defaults.

    Parameters
    ----------
    user_inputs : Dict[str, Any]
        Dictionary of user-provided values
    use_case : str
        One of: 'gaming', 'work', 'creative', 'student', 'general'

    Returns
    -------
    Dict[str, Any]
        Dictionary with all required fields filled
    """
    # Get profile for use case
    profile = USE_CASE_PROFILES.get(use_case, USE_CASE_PROFILES['general'])
    general = USE_CASE_PROFILES['general']

    # Start with general defaults, overlay use-case specific, then user inputs
    filled = {**general, **profile, **user_inputs}

    return filled


def get_cpu_benchmark_defaults(cpu_family: str) -> Dict[str, Any]:
    """
    Get default CPU benchmark values for a given CPU family.

    Parameters
    ----------
    cpu_family : str
        CPU family name (e.g., 'core i7', 'ryzen 7', 'm2')

    Returns
    -------
    Dict[str, Any]
        Dictionary with _cpu_mark, _cpu_rank, _cpu_value, _cpu_price_usd
    """
    cpu_family_lower = cpu_family.lower().strip()
    benchmarks = BENCHMARK_DEFAULTS['cpu_benchmarks']

    # Try exact match
    if cpu_family_lower in benchmarks:
        return benchmarks[cpu_family_lower]

    # Try partial match
    for key in benchmarks:
        if key in cpu_family_lower or cpu_family_lower in key:
            return benchmarks[key]

    return benchmarks['default']


def get_gpu_benchmark_defaults(gpu_type: str, is_integrated: bool = False) -> Dict[str, Any]:
    """
    Get default GPU benchmark values for a given GPU type.

    Parameters
    ----------
    gpu_type : str
        GPU type/series name (e.g., 'rtx 4060', 'rx 6600')
    is_integrated : bool
        Whether the GPU is integrated (returns None values)

    Returns
    -------
    Dict[str, Any]
        Dictionary with _gpu_mark, _gpu_rank, _gpu_value, _gpu_price_usd
    """
    if is_integrated:
        return BENCHMARK_DEFAULTS['gpu_benchmarks']['integrated']

    gpu_type_lower = gpu_type.lower().strip()
    benchmarks = BENCHMARK_DEFAULTS['gpu_benchmarks']

    # Try exact match
    if gpu_type_lower in benchmarks:
        return benchmarks[gpu_type_lower]

    # Try partial match
    for key in benchmarks:
        if key in gpu_type_lower or gpu_type_lower in key:
            return benchmarks[key]

    return benchmarks['default']


def get_confidence_level(user_inputs: Dict[str, Any],
                         required_fields: list = None) -> str:
    """
    Calculate confidence level based on how many fields user provided.

    Parameters
    ----------
    user_inputs : Dict[str, Any]
        User-provided values
    required_fields : list, optional
        List of important fields to check

    Returns
    -------
    str
        'high', 'medium', or 'low'
    """
    if required_fields is None:
        required_fields = [
            '_ram_gb', '_ssd_gb', 'cpu_family', 'gpu_series',
            '_tamano_pantalla_pulgadas', '_brand'
        ]

    provided = sum(1 for f in required_fields if f in user_inputs and user_inputs[f] is not None)
    total = len(required_fields)
    ratio = provided / total

    if ratio >= 0.8:
        return 'high'
    elif ratio >= 0.5:
        return 'medium'
    else:
        return 'low'
