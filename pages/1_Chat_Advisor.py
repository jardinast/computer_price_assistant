"""
Chat Advisor Page - AI-Powered Laptop Recommendation

This page provides a conversational interface where users can describe
their needs and get personalized laptop recommendations with price estimates.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Chat Advisor - Computer Price Predictor",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_advisor' not in st.session_state:
    st.session_state.chat_advisor = None
if 'show_initial_greeting' not in st.session_state:
    st.session_state.show_initial_greeting = True

# Custom CSS
st.markdown("""
<style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 1rem;
    }
    .quick-button {
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - API Key Configuration
with st.sidebar:
    st.header("⚙️ API Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for the Chat Advisor. Get your key at platform.openai.com",
        key="openai_api_key_chat"
    )

    if api_key:
        st.session_state['openai_api_key'] = api_key
        st.success("✅ API key configured!")

        # Initialize advisor if not done
        if st.session_state.chat_advisor is None:
            try:
                from src.llm_advisor import ChatAdvisor
                st.session_state.chat_advisor = ChatAdvisor(api_key)
            except Exception as e:
                st.error(f"Error initializing advisor: {e}")
    else:
        st.warning("⚠️ Enter your OpenAI API key to use the Chat Advisor")

    st.divider()

    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.show_initial_greeting = True
        if st.session_state.chat_advisor:
            st.session_state.chat_advisor.reset()
        st.rerun()

    st.divider()

    st.markdown("""
    ### How to Use

    1. Enter your OpenAI API key
    2. Tell me what you need
    3. Answer a few questions
    4. Get recommendations!

    ---

    **Example prompts:**
    - "I need a gaming laptop"
    - "Looking for a MacBook for work"
    - "Budget laptop for school"
    - "Laptop for video editing"
    """)

# Main content
st.title("🤖 Chat Advisor")
st.markdown("Tell me what you need, and I'll help you find the perfect laptop!")

# Check if API key is provided
if not st.session_state.get('openai_api_key'):
    st.info("""
    👈 **Please enter your OpenAI API key in the sidebar to start chatting.**

    Don't have an API key? Get one at [platform.openai.com](https://platform.openai.com)

    ---

    **In the meantime, you can use the Form Predictor:**
    """)

    if st.button("📝 Go to Form Predictor", type="primary"):
        st.switch_page("pages/2_Form_Predictor.py")

else:
    # Chat interface
    chat_container = st.container()

    with chat_container:
        # Show initial greeting
        if st.session_state.show_initial_greeting and not st.session_state.chat_messages:
            from src.llm_advisor import ChatAdvisor
            greeting = ChatAdvisor(st.session_state['openai_api_key']).get_initial_greeting()
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": greeting
            })
            st.session_state.show_initial_greeting = False

        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Quick action buttons (shown only at the start)
    if len(st.session_state.chat_messages) <= 1:
        st.markdown("**Quick Start:**")
        quick_cols = st.columns(4)

        quick_prompts = [
            ("🎮 Gaming Laptop", "I need a gaming laptop for playing AAA games"),
            ("💼 Work Laptop", "I need a laptop for work and productivity"),
            ("🎨 Creative Work", "I need a laptop for video editing and design"),
            ("📚 Student Budget", "I need an affordable laptop for school"),
        ]

        for col, (label, prompt) in zip(quick_cols, quick_prompts):
            with col:
                if st.button(label, use_container_width=True):
                    # Add as user message and process
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": prompt
                    })

                    # Get response
                    if st.session_state.chat_advisor:
                        with st.spinner("Thinking..."):
                            try:
                                response = st.session_state.chat_advisor.chat(prompt)
                                st.session_state.chat_messages.append({
                                    "role": "assistant",
                                    "content": response
                                })
                            except Exception as e:
                                st.error(f"Error: {e}")

                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Tell me about your laptop needs..."):
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            if st.session_state.chat_advisor:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chat_advisor.chat(prompt)
                        st.markdown(response)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response
                        })
                    except Exception as e:
                        error_msg = f"I encountered an error: {str(e)}. Please try again."
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
            else:
                st.error("Chat advisor not initialized. Please check your API key.")

# Footer
st.divider()
st.caption("""
**Note:** This advisor uses GPT-4 to understand your needs and provide recommendations.
Price predictions are estimates based on our trained model (±20% accuracy).
Your API key is used only for this session and is not stored.
""")
