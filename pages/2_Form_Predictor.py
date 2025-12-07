"""
Form Predictor Page - Simple and Advanced Mode Price Prediction

This page allows users to enter laptop specifications and get price predictions.
- Simple Mode: 6 key variables with smart defaults
- Advanced Mode: All 30 features for power users
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend_api import (
    predecir_precio,
    explicar_prediccion,
    obtener_campos_disponibles,
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
from components.price_display import (
    create_price_gauge,
    create_price_range_chart,
    create_feature_importance_chart,
    display_confidence_badge,
)

# Page config
st.set_page_config(
    page_title="Form Predictor - Computer Price Predictor",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'form_mode' not in st.session_state:
    st.session_state.form_mode = 'simple'

# Custom CSS
st.markdown("""
<style>
    .price-display {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
    }
    .price-range {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📝 Form Predictor")
st.markdown("Enter laptop specifications to get a price estimate")

# Mode toggle
col_toggle1, col_toggle2, col_toggle3 = st.columns([1, 2, 1])
with col_toggle2:
    mode = st.radio(
        "Mode",
        ["Simple", "Advanced"],
        horizontal=True,
        key="mode_selector",
        help="Simple mode uses smart defaults. Advanced mode shows all options."
    )
    st.session_state.form_mode = mode.lower()

st.divider()

# Main content - two columns
col_form, col_results = st.columns([3, 2])

with col_form:
    st.subheader("Specifications")

    if st.session_state.form_mode == 'simple':
        # Simple Mode - 6 key variables
        st.info("💡 Simple mode: Enter key specs, we'll fill in the rest based on your use case.")

        # Use Case
        use_case = st.selectbox(
            "What will you use this laptop for?",
            ["Gaming", "Work/Business", "Creative (Video/Photo)", "Student", "General Use"],
            index=4,
            help="This helps us set appropriate defaults for missing specs"
        )
        use_case_map = {
            "Gaming": "gaming",
            "Work/Business": "work",
            "Creative (Video/Photo)": "creative",
            "Student": "student",
            "General Use": "general"
        }
        use_case_key = use_case_map[use_case]

        # Brand
        brand = st.selectbox(
            "Brand",
            get_brand_options(),
            index=0,
            help="Laptop manufacturer"
        )

        # CPU - Simplified
        cpu_col1, cpu_col2 = st.columns(2)
        with cpu_col1:
            cpu_brand = st.selectbox(
                "CPU Brand",
                ["Intel", "AMD", "Apple"],
                index=0
            )
        with cpu_col2:
            cpu_families = get_cpu_family_options(cpu_brand)
            cpu_family = st.selectbox(
                "CPU Type",
                cpu_families,
                index=min(2, len(cpu_families) - 1)  # Default to i5/Ryzen 5/M2
            )

        # RAM and Storage
        mem_col1, mem_col2 = st.columns(2)
        with mem_col1:
            ram = st.select_slider(
                "RAM (GB)",
                options=[8, 16, 32, 64],
                value=16
            )
        with mem_col2:
            storage = st.select_slider(
                "Storage (GB)",
                options=[256, 512, 1000, 2000],
                value=512,
                format_func=lambda x: f"{x}GB" if x < 1000 else f"{x//1000}TB"
            )

        # GPU - Simplified
        gpu_type = st.selectbox(
            "Graphics",
            ["Integrated (Intel/AMD)", "NVIDIA RTX 4050", "NVIDIA RTX 4060",
             "NVIDIA RTX 4070", "NVIDIA RTX 4080", "NVIDIA RTX 3050",
             "NVIDIA RTX 3060", "AMD RX 6600", "AMD RX 7600"],
            index=0,
            help="Integrated is common for work/student laptops. Dedicated GPUs for gaming/creative."
        )

        # Map GPU type to internal values
        is_integrated = "Integrated" in gpu_type
        if is_integrated:
            gpu_series = "integrated"
            gpu_brand = "intel"
        elif "NVIDIA" in gpu_type:
            gpu_series = gpu_type.replace("NVIDIA ", "").lower()
            gpu_brand = "nvidia"
        else:
            gpu_series = gpu_type.replace("AMD ", "").lower()
            gpu_brand = "amd"

        # Build input dict for simple mode
        inputs = {
            '_brand': brand.lower(),
            'cpu_brand': cpu_brand.lower(),
            'cpu_family': cpu_family.lower(),
            '_ram_gb': ram,
            '_ssd_gb': storage,
            'gpu_brand': gpu_brand,
            'gpu_series': gpu_series,
            'gpu_is_integrated': is_integrated,
        }

    else:
        # Advanced Mode - All features
        st.info("🔧 Advanced mode: Fine-tune all specifications")

        # Use case still needed for any remaining defaults
        use_case = st.selectbox(
            "Primary Use Case",
            ["Gaming", "Work/Business", "Creative (Video/Photo)", "Student", "General Use"],
            index=4,
        )
        use_case_map = {
            "Gaming": "gaming",
            "Work/Business": "work",
            "Creative (Video/Photo)": "creative",
            "Student": "student",
            "General Use": "general"
        }
        use_case_key = use_case_map[use_case]

        st.markdown("#### Basic Info")
        brand = st.selectbox("Brand", get_brand_options(), index=0)

        st.markdown("#### CPU")
        cpu_col1, cpu_col2 = st.columns(2)
        with cpu_col1:
            cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD", "Apple", "Qualcomm"])
        with cpu_col2:
            cpu_families = get_cpu_family_options(cpu_brand)
            cpu_family = st.selectbox("CPU Family", cpu_families)

        cpu_col3, cpu_col4 = st.columns(2)
        with cpu_col3:
            cpu_cores = st.slider("CPU Cores", 2, 24, 8)

        st.markdown("#### GPU")
        gpu_col1, gpu_col2 = st.columns(2)
        with gpu_col1:
            is_integrated = st.checkbox("Integrated Graphics", value=False)
        with gpu_col2:
            if not is_integrated:
                gpu_brand = st.selectbox("GPU Brand", ["NVIDIA", "AMD", "Intel Arc"])
            else:
                gpu_brand = "intel"

        if not is_integrated:
            gpu_col3, gpu_col4 = st.columns(2)
            with gpu_col3:
                gpu_options = get_gpu_series_options(gpu_brand)
                gpu_series = st.selectbox("GPU Model", gpu_options)
            with gpu_col4:
                gpu_memory = st.slider("GPU Memory (GB)", 2, 24, 6)
        else:
            gpu_series = "integrated"
            gpu_memory = 0

        st.markdown("#### Memory & Storage")
        mem_col1, mem_col2 = st.columns(2)
        with mem_col1:
            ram = st.select_slider("RAM (GB)", options=get_ram_options(), value=16)
        with mem_col2:
            storage = st.select_slider(
                "Storage (GB)",
                options=get_storage_options(),
                value=512,
                format_func=lambda x: f"{x}GB" if x < 1000 else f"{x//1000}TB"
            )

        st.markdown("#### Display")
        disp_col1, disp_col2, disp_col3 = st.columns(3)
        with disp_col1:
            screen_size = st.slider("Screen Size (inches)", 11.0, 18.0, 15.6, 0.1)
        with disp_col2:
            resolution_options = get_resolution_options()
            resolution_name = st.selectbox("Resolution", list(resolution_options.keys()), index=2)
            resolution = resolution_options[resolution_name]
        with disp_col3:
            refresh_rate = st.select_slider(
                "Refresh Rate (Hz)",
                options=get_refresh_rate_options(),
                value=60
            )

        st.markdown("#### Physical")
        phys_col1, phys_col2 = st.columns(2)
        with phys_col1:
            weight = st.slider("Weight (kg)", 0.5, 4.5, 1.8, 0.1)

        st.markdown("#### Features")
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        with feat_col1:
            has_wifi = st.checkbox("WiFi", value=True)
        with feat_col2:
            has_bluetooth = st.checkbox("Bluetooth", value=True)
        with feat_col3:
            has_webcam = st.checkbox("Webcam", value=True)

        # Build input dict for advanced mode
        inputs = {
            '_brand': brand.lower(),
            'cpu_brand': cpu_brand.lower(),
            'cpu_family': cpu_family.lower(),
            '_cpu_cores': cpu_cores,
            '_ram_gb': ram,
            '_ssd_gb': storage,
            'gpu_brand': gpu_brand.lower() if not is_integrated else 'intel',
            'gpu_series': gpu_series.lower() if not is_integrated else 'integrated',
            'gpu_is_integrated': is_integrated,
            '_gpu_memory_gb': gpu_memory if not is_integrated else 0,
            '_tamano_pantalla_pulgadas': screen_size,
            '_resolucion_pixeles': resolution,
            '_tasa_refresco_hz': refresh_rate,
            '_peso_kg': weight,
            '_tiene_wifi': 1 if has_wifi else 0,
            '_tiene_bluetooth': 1 if has_bluetooth else 0,
            '_tiene_webcam': 1 if has_webcam else 0,
        }

    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner("Calculating price..."):
            try:
                result = explicar_prediccion(inputs, use_case=use_case_key)
                st.session_state.prediction_result = result
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.session_state.prediction_result = None

# Results column
with col_results:
    st.subheader("Price Estimate")

    if st.session_state.prediction_result:
        result = st.session_state.prediction_result

        # Price gauge visualization
        gauge_fig = create_price_gauge(
            result['prediccion'],
            min_price=0,
            max_price=3605
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Price range bar
        range_fig = create_price_range_chart(
            result['prediccion'],
            result['prediccion_min'],
            result['prediccion_max']
        )
        st.plotly_chart(range_fig, use_container_width=True)

        # Confidence indicator
        st.markdown("<br>", unsafe_allow_html=True)
        display_confidence_badge(result['confidence'])

        st.divider()

        # Feature importance chart
        st.subheader("What's Driving This Price?")

        importance_fig = create_feature_importance_chart(result['top_features'], max_features=7)
        st.plotly_chart(importance_fig, use_container_width=True)

        # Explanation text
        st.info(result['explanation_text'])

        # Additional insights
        with st.expander("See all feature importances"):
            for feat in result['top_features']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(feat['readable_name'])
                with col2:
                    st.write(f"{feat['importance_pct']:.1f}%")

    else:
        # Placeholder
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px;">
            <p style="font-size: 1.2rem; color: #666;">
                Enter your specifications and click<br>
                <strong>"Predict Price"</strong><br>
                to see the estimate
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer info
st.divider()
st.caption("""
**Note:** Price predictions are estimates based on a machine learning model trained on laptop listings.
Actual prices may vary based on retailer, availability, and market conditions.
Model accuracy: approximately ±20% (MAPE).
""")
