"""
Market Overview Page - Descriptive Analytics

This page provides:
- Statistical overview/summary of the market offerings
- Clustering/segmentation of different types of products
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
    load_computers_data,
    get_market_statistics,
    get_price_by_category,
    get_clustering_data,
    get_cluster_profiles,
)

# Page config
st.set_page_config(
    page_title="Market Overview - Computer Price Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border-left: 4px solid #4F8BF9;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .segment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Market Overview")
st.markdown("Explore the laptop market with statistics, trends, and segmentation analysis")

# Tabs for different views
tab1, tab2 = st.tabs(["📈 Statistics", "🎯 Segmentation"])

# =============================================================================
# TAB 1: STATISTICAL OVERVIEW
# =============================================================================
with tab1:
    st.header("Market Statistics")

    # Load statistics
    with st.spinner("Loading market data..."):
        stats = get_market_statistics()
        df = load_computers_data()

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Listings",
            f"{stats['total_listings']:,}",
            help="Total number of laptop listings in the database"
        )

    with col2:
        if stats['price_stats']:
            st.metric(
                "Average Price",
                f"€{stats['price_stats']['mean']:,.0f}",
                help="Mean price across all listings"
            )

    with col3:
        if stats['price_stats']:
            st.metric(
                "Median Price",
                f"€{stats['price_stats']['median']:,.0f}",
                help="Median price (50th percentile)"
            )

    with col4:
        if stats['price_stats']:
            st.metric(
                "Price Range",
                f"€{stats['price_stats']['min']:,.0f} - €{stats['price_stats']['max']:,.0f}",
                help="Min to max price range"
            )

    st.divider()

    # Charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Price Distribution")
        if '_precio_medio' in df.columns:
            price_data = df['_precio_medio'].dropna()
            fig = px.histogram(
                price_data,
                nbins=50,
                title="",
                labels={'value': 'Price (EUR)', 'count': 'Number of Laptops'},
                color_discrete_sequence=['#4F8BF9']
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Price (EUR)",
                yaxis_title="Count",
                height=350
            )
            fig.add_vline(
                x=price_data.median(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: €{price_data.median():,.0f}"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top Brands by Listings")
        if stats['brand_distribution']:
            brand_df = pd.DataFrame([
                {'Brand': k, 'Count': v}
                for k, v in stats['brand_distribution'].items()
            ])
            fig = px.bar(
                brand_df,
                x='Count',
                y='Brand',
                orientation='h',
                color='Count',
                color_continuous_scale='Blues',
                title=""
            )
            fig.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                height=350,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("CPU Brand Distribution")
        if stats['cpu_distribution']:
            cpu_df = pd.DataFrame([
                {'CPU': k, 'Count': v}
                for k, v in stats['cpu_distribution'].items()
            ])
            fig = px.pie(
                cpu_df,
                values='Count',
                names='CPU',
                title="",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        st.subheader("GPU Brand Distribution")
        if stats['gpu_distribution']:
            gpu_df = pd.DataFrame([
                {'GPU': k, 'Count': v}
                for k, v in stats['gpu_distribution'].items()
            ])
            fig = px.pie(
                gpu_df,
                values='Count',
                names='GPU',
                title="",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Charts row 3
    col_left3, col_right3 = st.columns(2)

    with col_left3:
        st.subheader("RAM Distribution")
        if stats['ram_distribution']:
            ram_df = pd.DataFrame([
                {'RAM': k, 'Count': v}
                for k, v in stats['ram_distribution'].items()
            ])
            fig = px.bar(
                ram_df,
                x='RAM',
                y='Count',
                color='Count',
                color_continuous_scale='Greens',
                title=""
            )
            fig.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_right3:
        st.subheader("Storage Distribution")
        if stats['storage_distribution']:
            storage_df = pd.DataFrame([
                {'Storage': k, 'Count': v}
                for k, v in stats['storage_distribution'].items()
            ])
            fig = px.bar(
                storage_df,
                x='Storage',
                y='Count',
                color='Count',
                color_continuous_scale='Oranges',
                title=""
            )
            fig.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Price by category
    st.divider()
    st.subheader("Price Analysis by Category")

    category = st.selectbox(
        "Analyze prices by:",
        ["brand", "cpu", "gpu"],
        format_func=lambda x: {"brand": "Brand", "cpu": "CPU Manufacturer", "gpu": "GPU Manufacturer"}[x]
    )

    price_by_cat = get_price_by_category(category)
    if not price_by_cat.empty:
        # Show top 15
        price_by_cat = price_by_cat.head(15)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Mean Price',
            x=price_by_cat.iloc[:, 0],
            y=price_by_cat['Mean Price'],
            marker_color='#4F8BF9'
        ))
        fig.add_trace(go.Bar(
            name='Median Price',
            x=price_by_cat.iloc[:, 0],
            y=price_by_cat['Median Price'],
            marker_color='#28a745'
        ))
        fig.update_layout(
            barmode='group',
            xaxis_title=category.title(),
            yaxis_title='Price (EUR)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("View detailed data"):
            st.dataframe(
                price_by_cat.style.format({
                    'Mean Price': '€{:,.0f}',
                    'Median Price': '€{:,.0f}',
                    'Min Price': '€{:,.0f}',
                    'Max Price': '€{:,.0f}',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )


# =============================================================================
# TAB 2: CLUSTERING/SEGMENTATION
# =============================================================================
with tab2:
    st.header("Market Segmentation")
    st.markdown("Laptops clustered into segments based on price, RAM, storage, and screen size")

    with st.spinner("Performing cluster analysis..."):
        df_clustered, X_pca, clusters = get_clustering_data()
        profiles = get_cluster_profiles()

    if len(profiles) > 0:
        # Segment overview cards
        st.subheader("Market Segments")

        cols = st.columns(len(profiles))
        for i, (col, profile) in enumerate(zip(cols, profiles)):
            with col:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg,
                        {'#667eea' if i == 0 else '#f093fb' if i == 1 else '#4facfe' if i == 2 else '#43e97b' if i == 3 else '#fa709a'} 0%,
                        {'#764ba2' if i == 0 else '#f5576c' if i == 1 else '#00f2fe' if i == 2 else '#38f9d7' if i == 3 else '#fee140'} 100%);
                    border-radius: 10px;
                    padding: 1rem;
                    color: white;
                    text-align: center;
                    min-height: 180px;
                ">
                    <h4 style="margin: 0; color: white;">{profile['segment_name']}</h4>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">€{profile['avg_price']:,.0f}</p>
                    <p style="font-size: 0.9rem; margin: 0;">{profile['count']:,} laptops ({profile['percentage']:.1f}%)</p>
                    <p style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.9;">{profile['description']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Cluster visualization
        col_viz, col_details = st.columns([2, 1])

        with col_viz:
            st.subheader("Cluster Visualization (PCA)")

            if len(X_pca) > 0:
                # Create scatter plot
                viz_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': clusters.astype(str)
                })

                fig = px.scatter(
                    viz_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title="",
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    opacity=0.6
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for visualization")

        with col_details:
            st.subheader("Segment Details")

            for profile in profiles:
                with st.expander(f"{profile['segment_name']} ({profile['count']:,} laptops)"):
                    st.markdown(f"""
                    **Average Specifications:**
                    - Price: €{profile['avg_price']:,.0f}
                    - RAM: {profile['avg_ram']:.0f} GB
                    - Storage: {profile['avg_storage']:.0f} GB
                    - Screen: {profile['avg_screen']:.1f}"
                    - Top Brand: {profile['top_brand']}

                    {profile['description']}
                    """)

        # Segment comparison chart
        st.subheader("Segment Comparison")

        comparison_data = pd.DataFrame(profiles)

        fig = go.Figure()

        # Normalize values for radar chart
        metrics = ['avg_price', 'avg_ram', 'avg_storage', 'avg_screen']
        metric_labels = ['Avg Price', 'Avg RAM (GB)', 'Avg Storage (GB)', 'Avg Screen Size']

        for profile in profiles:
            fig.add_trace(go.Scatterpolar(
                r=[
                    profile['avg_price'] / 100,  # Scale down price
                    profile['avg_ram'],
                    profile['avg_storage'] / 50,  # Scale down storage
                    profile['avg_screen']
                ],
                theta=metric_labels,
                fill='toself',
                name=profile['segment_name']
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 40])
            ),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Unable to perform clustering. Please ensure data is available.")

# Footer
st.divider()
st.caption("""
**Note:** Statistics are based on the laptop listings in our database.
Market conditions and actual availability may vary.
""")
