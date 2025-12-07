"""
Find Similar Laptops Page - Prescriptive Analytics

This page provides:
- K-best offers similar to requested features
- Distance mapping in terms of characteristics, predicted price, and real price
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics import (
    find_similar_laptops,
    get_sample_laptops,
    load_computers_data,
)
from src.backend_api import predecir_precio
from src.benchmark_cache import get_brand_options

# Page config
st.set_page_config(
    page_title="Find Similar Laptops - Computer Price Predictor",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .laptop-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4F8BF9;
    }
    .similarity-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .price-tag {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .spec-chip {
        background-color: #e9ecef;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        margin-right: 0.5rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 Find Similar Laptops")
st.markdown("Discover laptops that match your requirements based on specifications and price")

# Sidebar for filters
with st.sidebar:
    st.header("Search Criteria")

    st.markdown("### Budget")
    price_target = st.slider(
        "Target Price (EUR)",
        min_value=200,
        max_value=3500,
        value=1000,
        step=50,
        help="Your ideal price point"
    )

    price_range = st.slider(
        "Price Flexibility (%)",
        min_value=0,
        max_value=50,
        value=20,
        help="How much above/below your target price to search"
    )

    st.markdown("### Specifications")

    ram_gb = st.select_slider(
        "RAM (GB)",
        options=[4, 8, 16, 32, 64, 128],
        value=16
    )

    ssd_gb = st.select_slider(
        "Storage (GB)",
        options=[128, 256, 512, 1000, 2000],
        value=512,
        format_func=lambda x: f"{x}GB" if x < 1000 else f"{x//1000}TB"
    )

    screen_size = st.slider(
        "Screen Size (inches)",
        min_value=11.0,
        max_value=18.0,
        value=15.6,
        step=0.1
    )

    st.markdown("### Preferences (Optional)")

    brand_options = ['Any'] + get_brand_options()
    brand = st.selectbox("Preferred Brand", brand_options)

    cpu_brand = st.selectbox(
        "CPU Brand",
        ['Any', 'Intel', 'AMD', 'Apple']
    )

    st.markdown("### Results")
    num_results = st.slider(
        "Number of results",
        min_value=3,
        max_value=20,
        value=5
    )

# Build search specs
specs = {
    'price_target': price_target,
    'ram_gb': ram_gb,
    'ssd_gb': ssd_gb,
    'screen_size': screen_size,
}

if brand != 'Any':
    specs['brand'] = brand
if cpu_brand != 'Any':
    specs['cpu_brand'] = cpu_brand

# Main content
col_main, col_summary = st.columns([3, 1])

with col_summary:
    st.markdown("### Your Requirements")

    st.markdown(f"""
    <div style="background-color: #f0f7ff; padding: 1rem; border-radius: 10px;">
        <p><strong>Target Price:</strong> €{price_target:,}</p>
        <p><strong>RAM:</strong> {ram_gb} GB</p>
        <p><strong>Storage:</strong> {ssd_gb if ssd_gb < 1000 else f'{ssd_gb//1000}TB'}</p>
        <p><strong>Screen:</strong> {screen_size}"</p>
        <p><strong>Brand:</strong> {brand}</p>
        <p><strong>CPU:</strong> {cpu_brand}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Get predicted price for these specs
    try:
        predicted_inputs = {
            '_ram_gb': ram_gb,
            '_ssd_gb': ssd_gb,
            '_tamano_pantalla_pulgadas': screen_size,
            'cpu_brand': cpu_brand.lower() if cpu_brand != 'Any' else 'intel',
            '_brand': brand.lower() if brand != 'Any' else 'asus',
        }
        predicted_price = predecir_precio(predicted_inputs, use_case='general')

        st.markdown("### Predicted Price")
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #e8f4ea; border-radius: 10px;">
            <p style="font-size: 2rem; font-weight: bold; color: #28a745; margin: 0;">
                €{predicted_price:,.0f}
            </p>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">
                Model estimate for your specs
            </p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.info("Enter specs to see predicted price")

with col_main:
    st.markdown("### Similar Laptops")

    # Find similar laptops
    with st.spinner("Searching for similar laptops..."):
        results = find_similar_laptops(specs, k=num_results)

    if results.empty:
        st.warning("No similar laptops found. Try adjusting your criteria.")

        # Show sample laptops instead
        st.markdown("### Browse Available Laptops")
        samples = get_sample_laptops(10)
        if not samples.empty:
            st.dataframe(samples, use_container_width=True)
    else:
        # Display results as cards
        for idx, row in results.iterrows():
            # Calculate similarity score (inverse of distance, normalized)
            max_dist = results['_similarity_distance'].max()
            min_dist = results['_similarity_distance'].min()
            if max_dist > min_dist:
                similarity_score = 100 * (1 - (row['_similarity_distance'] - min_dist) / (max_dist - min_dist + 0.01))
            else:
                similarity_score = 100

            # Determine match quality
            if similarity_score >= 80:
                match_color = '#28a745'
                match_text = 'Excellent Match'
            elif similarity_score >= 60:
                match_color = '#17a2b8'
                match_text = 'Good Match'
            elif similarity_score >= 40:
                match_color = '#ffc107'
                match_text = 'Fair Match'
            else:
                match_color = '#dc3545'
                match_text = 'Partial Match'

            st.markdown(f"""
            <div class="laptop-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 3;">
                        <h4 style="margin: 0 0 0.5rem 0;">#{int(row['_similarity_rank'])}. {row.get('Título', 'Unknown Laptop')[:80]}...</h4>
                        <div style="margin-bottom: 0.5rem;">
                            <span class="spec-chip">RAM: {row.get('_ram_gb', 'N/A')} GB</span>
                            <span class="spec-chip">Storage: {row.get('_ssd_gb', 'N/A')} GB</span>
                            <span class="spec-chip">Screen: {row.get('_screen_size', 'N/A'):.1f}"</span>
                            <span class="spec-chip">{row.get('_cpu_brand', 'N/A')}</span>
                        </div>
                        <p style="color: #666; font-size: 0.9rem; margin: 0;">
                            Brand: {row.get('_brand', 'Unknown')} |
                            GPU: {row.get('_gpu_brand', 'Unknown')}
                        </p>
                    </div>
                    <div style="flex: 1; text-align: right;">
                        <p class="price-tag">€{row.get('_precio_medio', 0):,.0f}</p>
                        <p style="font-size: 0.85rem; color: #666;">
                            Range: €{row.get('_precio_min', 0):,.0f} - €{row.get('_precio_max', 0):,.0f}
                        </p>
                        <span style="background-color: {match_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem;">
                            {match_text}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Comparison visualization
        st.subheader("How They Compare")

        # Create comparison chart
        compare_data = []
        for idx, row in results.iterrows():
            compare_data.append({
                'Laptop': f"#{int(row['_similarity_rank'])}",
                'Price': row.get('_precio_medio', 0),
                'RAM': row.get('_ram_gb', 0),
                'Storage': row.get('_ssd_gb', 0) / 100,  # Scale for visibility
                'Screen': row.get('_screen_size', 0),
            })

        # Add user target
        compare_data.append({
            'Laptop': 'Your Target',
            'Price': price_target,
            'RAM': ram_gb,
            'Storage': ssd_gb / 100,
            'Screen': screen_size,
        })

        compare_df = pd.DataFrame(compare_data)

        # Radar chart
        fig = go.Figure()

        for i, row in compare_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[
                    row['Price'] / 100,  # Scale
                    row['RAM'],
                    row['Storage'],
                    row['Screen']
                ],
                theta=['Price (€/100)', 'RAM (GB)', 'Storage (GB/100)', 'Screen (inches)'],
                fill='toself',
                name=row['Laptop'],
                opacity=0.7 if row['Laptop'] != 'Your Target' else 1,
                line=dict(width=3) if row['Laptop'] == 'Your Target' else dict(width=1)
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 35])),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Price comparison bar chart
        st.subheader("Price Comparison")

        results_for_chart = results.copy()
        results_for_chart['Label'] = results_for_chart.apply(
            lambda x: f"#{int(x['_similarity_rank'])} {x['_brand']}",
            axis=1
        )

        fig2 = go.Figure()

        # Actual prices
        fig2.add_trace(go.Bar(
            name='Actual Price',
            x=results_for_chart['Label'],
            y=results_for_chart['_precio_medio'],
            marker_color='#4F8BF9',
            text=[f"€{p:,.0f}" for p in results_for_chart['_precio_medio']],
            textposition='outside'
        ))

        # Add target line
        fig2.add_hline(
            y=price_target,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Your Target: €{price_target:,}"
        )

        fig2.update_layout(
            xaxis_title="Laptop",
            yaxis_title="Price (EUR)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Detailed data table
        with st.expander("View All Details"):
            display_cols = ['_similarity_rank', 'Título', '_precio_medio', '_brand',
                           '_ram_gb', '_ssd_gb', '_screen_size', '_cpu_brand', '_gpu_brand']
            display_cols = [c for c in display_cols if c in results.columns]

            display_df = results[display_cols].copy()
            display_df.columns = ['Rank', 'Title', 'Price (€)', 'Brand', 'RAM (GB)',
                                  'Storage (GB)', 'Screen Size', 'CPU', 'GPU'][:len(display_cols)]

            st.dataframe(
                display_df.style.format({
                    'Price (€)': '€{:,.0f}',
                    'RAM (GB)': '{:.0f}',
                    'Storage (GB)': '{:.0f}',
                    'Screen Size': '{:.1f}"'
                }),
                use_container_width=True
            )

# Footer
st.divider()
st.caption("""
**Note:** Similarity is calculated based on price, RAM, storage, and screen size.
Results show laptops from our database that most closely match your requirements.
Prices are estimates and may vary by retailer.
""")
