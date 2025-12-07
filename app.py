"""
Computer Price Predictor - Main Application
A multi-page web application for laptop price prediction and market analysis.

Features:
1. Descriptive: Market statistics and clustering/segmentation
2. Predictive: Price prediction with explanatory breakdown
3. Prescriptive: Similar laptop recommendations (k-best offers)
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Computer Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4F8BF9;
        min-height: 200px;
    }
    .feature-card-descriptive { border-left-color: #28a745; }
    .feature-card-predictive { border-left-color: #4F8BF9; }
    .feature-card-prescriptive { border-left-color: #ffc107; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .category-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - API Key Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for the Chat Advisor. Get your key at platform.openai.com",
        key="openai_api_key_input"
    )

    if api_key:
        st.session_state['openai_api_key'] = api_key
        st.success("✅ API key configured!")
    else:
        st.info("💡 Enter API key to use Chat Advisor")

    st.divider()

    st.markdown("""
    ### About This App

    A comprehensive laptop analysis tool with:

    **1. Descriptive Analytics**
    - Market statistics & trends
    - Product segmentation

    **2. Predictive Analytics**
    - Price prediction
    - Feature importance

    **3. Prescriptive Analytics**
    - Similar laptop finder
    - Best offers comparison

    ---

    **Model Info:**
    - Algorithm: CatBoost
    - Accuracy: ±20% (MAPE)
    - Data: 5,915 laptop listings
    """)

# Main content
st.markdown('<p class="main-header">💻 Computer Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore the laptop market, predict prices, and find your perfect match</p>', unsafe_allow_html=True)

# Section 1: Descriptive Analytics
st.markdown("### 📊 Descriptive Analytics")
st.markdown("*Understand the laptop market through data*")

desc_col1, desc_col2 = st.columns(2)

with desc_col1:
    st.markdown("""
    <div class="feature-card feature-card-descriptive">
        <p class="category-label" style="color: #28a745;">DESCRIPTIVE</p>
        <h3>📈 Market Overview</h3>
        <p>Statistical summary and trends of the laptop market:</p>
        <ul>
            <li>Price distributions by brand, CPU, GPU</li>
            <li>Market share analysis</li>
            <li>Feature correlations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📊 View Statistics", key="stats_btn", type="secondary"):
        st.switch_page("pages/3_Market_Overview.py")

with desc_col2:
    st.markdown("""
    <div class="feature-card feature-card-descriptive">
        <p class="category-label" style="color: #28a745;">DESCRIPTIVE</p>
        <h3>🎯 Market Segmentation</h3>
        <p>Clustering analysis of different laptop types:</p>
        <ul>
            <li>5 market segments identified</li>
            <li>Segment profiles & characteristics</li>
            <li>Visual cluster mapping</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🎯 Explore Segments", key="segment_btn", type="secondary"):
        st.switch_page("pages/3_Market_Overview.py")

st.divider()

# Section 2: Predictive Analytics
st.markdown("### 🔮 Predictive Analytics")
st.markdown("*Get price estimates and understand what drives laptop prices*")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.markdown("""
    <div class="feature-card feature-card-predictive">
        <p class="category-label" style="color: #4F8BF9;">PREDICTIVE</p>
        <h3>🤖 Chat Advisor</h3>
        <p>AI-powered conversational assistant:</p>
        <ul>
            <li>Tell us your needs naturally</li>
            <li>Get personalized recommendations</li>
            <li>Budget-aware suggestions</li>
        </ul>
        <p><em>Requires OpenAI API key</em></p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗣️ Start Chatting", key="chat_btn", type="primary"):
        st.switch_page("pages/1_Chat_Advisor.py")

with pred_col2:
    st.markdown("""
    <div class="feature-card feature-card-predictive">
        <p class="category-label" style="color: #4F8BF9;">PREDICTIVE</p>
        <h3>📝 Form Predictor</h3>
        <p>Direct specification input with price breakdown:</p>
        <ul>
            <li>Simple mode (6 key specs)</li>
            <li>Advanced mode (30+ features)</li>
            <li>Feature importance visualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📋 Use Form", key="form_btn", type="primary"):
        st.switch_page("pages/2_Form_Predictor.py")

st.divider()

# Section 3: Prescriptive Analytics
st.markdown("### 🔍 Prescriptive Analytics")
st.markdown("*Find the best laptops matching your requirements*")

presc_col1, presc_col2 = st.columns(2)

with presc_col1:
    st.markdown("""
    <div class="feature-card feature-card-prescriptive">
        <p class="category-label" style="color: #ffc107;">PRESCRIPTIVE</p>
        <h3>🔍 Find Similar Laptops</h3>
        <p>Similarity search for best matching offers:</p>
        <ul>
            <li>K-best offers based on your specs</li>
            <li>Distance mapping (specs vs price)</li>
            <li>Predicted vs actual price comparison</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Find Matches", key="similar_btn", type="secondary"):
        st.switch_page("pages/4_Find_Similar.py")

with presc_col2:
    st.markdown("""
    <div class="feature-card">
        <p class="category-label" style="color: #666;">QUICK START</p>
        <h3>💡 Example Use Cases</h3>
        <table style="width: 100%; font-size: 0.9rem;">
            <tr><td><strong>🎮 Gaming</strong></td><td>RTX 4060, i7, 16GB</td><td>€1,200-1,500</td></tr>
            <tr><td><strong>💼 Work</strong></td><td>i5, Integrated, 16GB</td><td>€700-900</td></tr>
            <tr><td><strong>🎨 Creative</strong></td><td>M3 Pro, 32GB</td><td>€2,000-2,500</td></tr>
            <tr><td><strong>📚 Student</strong></td><td>i5, 8GB, 256GB</td><td>€500-700</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("""
    **Navigation**
    - [Chat Advisor](pages/1_Chat_Advisor.py) - AI recommendations
    - [Form Predictor](pages/2_Form_Predictor.py) - Manual input
    """)

with col_footer2:
    st.markdown("""
    **Analytics**
    - [Market Overview](pages/3_Market_Overview.py) - Statistics
    - [Find Similar](pages/4_Find_Similar.py) - Search
    """)

with col_footer3:
    st.markdown("""
    **Info**
    - Built for DAI Assignment 2
    - Model: CatBoost Regressor
    - Data: 5,915 laptops
    """)

st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; margin-top: 1rem;">
    Computer Price Predictor | DAI 2025 | University Assignment
</div>
""", unsafe_allow_html=True)
