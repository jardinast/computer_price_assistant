"""
Price Display Components - Visualizations for price predictions
"""

import plotly.graph_objects as go
import streamlit as st


def create_price_gauge(price: float, min_price: float = 0, max_price: float = 3605) -> go.Figure:
    """
    Create a gauge chart showing the predicted price.

    Parameters
    ----------
    price : float
        The predicted price
    min_price : float
        Minimum price in the model's range
    max_price : float
        Maximum price in the model's range

    Returns
    -------
    go.Figure
        Plotly gauge figure
    """
    # Define price zones
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'prefix': "€", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [min_price, max_price], 'tickprefix': '€'},
            'bar': {'color': "#4F8BF9"},
            'steps': [
                {'range': [0, 600], 'color': "#e8f4ea"},
                {'range': [600, 1200], 'color': "#d4edda"},
                {'range': [1200, 2000], 'color': "#fff3cd"},
                {'range': [2000, 2800], 'color': "#ffe5d0"},
                {'range': [2800, 3605], 'color': "#f8d7da"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': price
            }
        },
        title={'text': "Estimated Price", 'font': {'size': 16}}
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def create_price_range_chart(price: float, price_min: float, price_max: float) -> go.Figure:
    """
    Create a horizontal bar showing price range with confidence interval.

    Parameters
    ----------
    price : float
        The predicted price
    price_min : float
        Lower bound of prediction
    price_max : float
        Upper bound of prediction

    Returns
    -------
    go.Figure
        Plotly figure with price range visualization
    """
    fig = go.Figure()

    # Add the range bar
    fig.add_trace(go.Bar(
        x=[price_max - price_min],
        y=['Price Range'],
        orientation='h',
        base=price_min,
        marker_color='rgba(79, 139, 249, 0.3)',
        name='Confidence Range',
        hovertemplate=f'Range: €{price_min:,.0f} - €{price_max:,.0f}<extra></extra>'
    ))

    # Add the predicted price marker
    fig.add_trace(go.Scatter(
        x=[price],
        y=['Price Range'],
        mode='markers',
        marker=dict(size=20, color='#4F8BF9', symbol='diamond'),
        name='Predicted',
        hovertemplate=f'Predicted: €{price:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(
            title='Price (EUR)',
            tickprefix='€',
            range=[max(0, price_min - 100), price_max + 100]
        ),
        yaxis=dict(visible=False),
        bargap=0.5,
    )

    return fig


def create_feature_importance_chart(top_features: list, max_features: int = 8) -> go.Figure:
    """
    Create a horizontal bar chart for feature importances.

    Parameters
    ----------
    top_features : list
        List of feature dicts with 'readable_name' and 'importance_pct'
    max_features : int
        Maximum number of features to show

    Returns
    -------
    go.Figure
        Plotly horizontal bar chart
    """
    features = top_features[:max_features]

    # Reverse for correct display order
    names = [f['readable_name'] for f in features][::-1]
    values = [f['importance_pct'] for f in features][::-1]

    # Color gradient from light to dark blue
    colors = [f'rgba(79, 139, 249, {0.4 + 0.6 * (i / len(values))})' for i in range(len(values))]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        height=max(200, 35 * len(features)),
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            title='Importance (%)',
            range=[0, max(values) * 1.2]
        ),
        yaxis=dict(title=''),
        showlegend=False,
    )

    return fig


def display_confidence_badge(confidence: str):
    """
    Display a confidence badge with appropriate styling.

    Parameters
    ----------
    confidence : str
        'high', 'medium', or 'low'
    """
    colors = {
        'high': ('#28a745', '#d4edda'),
        'medium': ('#856404', '#fff3cd'),
        'low': ('#721c24', '#f8d7da')
    }
    icons = {
        'high': '✓',
        'medium': '~',
        'low': '!'
    }

    text_color, bg_color = colors.get(confidence, colors['medium'])
    icon = icons.get(confidence, '~')

    st.markdown(f"""
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 0.5rem 1rem;
            border-radius: 20px;
            text-align: center;
            font-weight: 600;
        ">
            {icon} Confidence: {confidence.title()}
        </div>
    """, unsafe_allow_html=True)


def display_specs_summary(specs: dict):
    """
    Display a formatted summary of laptop specifications.

    Parameters
    ----------
    specs : dict
        Dictionary of specification values
    """
    spec_items = []

    if 'cpu_family' in specs:
        spec_items.append(f"**CPU:** {specs['cpu_family'].title()}")

    if '_ram_gb' in specs:
        spec_items.append(f"**RAM:** {specs['_ram_gb']}GB")

    if '_ssd_gb' in specs:
        storage = specs['_ssd_gb']
        storage_str = f"{storage}GB" if storage < 1000 else f"{storage//1000}TB"
        spec_items.append(f"**Storage:** {storage_str}")

    if 'gpu_series' in specs:
        gpu = specs['gpu_series']
        if gpu == 'integrated':
            spec_items.append("**GPU:** Integrated")
        else:
            spec_items.append(f"**GPU:** {gpu.upper()}")

    if '_brand' in specs:
        spec_items.append(f"**Brand:** {specs['_brand'].title()}")

    st.markdown(" | ".join(spec_items))
