import openai
import streamlit as st
import pandas as pd
import os
import random
from datetime import datetime

# Try to get the OpenAI API key from different sources
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("⚠️ OpenAI API key not found!")
    st.stop()

# Load the dataset
data_path = "agentic_ai_performance_dataset_20250622.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Dataset file not found: {data_path}")
    st.stop()

# Streamlit app setup
st.set_page_config(
    page_title="AI Agent Hub",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    body {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
        color: #1a1f36;
    }

    /* App Container */
    .stApp {
        background: linear-gradient(145deg, #f3f6ff 0%, #ffffff 100%);
    }

    /* Header/Title Section */
    .header-container {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #2b6cb0 0%, #1e4e8c 100%);
        border-radius: 24px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }
    .app-title {
        font-size: 2.8em;
        font-weight: 700;
        margin: 0;
        padding: 0;
        background: white;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .app-subtitle {
        font-size: 1.2em;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Chat Interface */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        padding: 2rem;
        margin: 1.5rem auto;
        max-width: 800px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        animation: slideUp 0.5s ease-out;
    }

    /* Messages */
    .message {
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease-out;
        position: relative;
    }
    .user-message {
        background: linear-gradient(135deg, #2b6cb0 0%, #1e4e8c 100%);
        color: white;
        border-radius: 20px 20px 5px 20px;
        padding: 1rem 1.5rem;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(43, 108, 176, 0.2);
    }
    .agent-message {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 20px 20px 20px 5px;
        padding: 1rem 1.5rem;
        margin-right: 20%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    /* Quick Actions */
    .quick-actions {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
        justify-content: center;
    }
    .action-button {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        min-width: 120px;
    }
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: #2b6cb0;
    }

    /* Input Area */
    .input-area {
        display: flex;
        gap: 1rem;
        margin-top: 2rem;
        background: white;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2b6cb0 !important;
        box-shadow: 0 0 0 3px rgba(43, 108, 176, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #1e4e8c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: auto !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(43, 108, 176, 0.2) !important;
    }

    /* Animations */
    @keyframes slideUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Recommendation Cards */
    .recommendation-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* Feature Tags */
    .feature-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9em;
        margin: 0.2rem;
        background: #f1f5f9;
        color: #1e4e8c;
    }
    
    /* Settings Panel */
    .settings-panel {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="app-title">AI Agent Hub 🤖</h1>
        <p class="app-subtitle">Your Intelligent Agent Advisor</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'preferences' not in st.session_state:
    st.session_state.preferences = {
        'task_complexity': 5,
        'autonomy_level': 5,
        'task_category': 'general'
    }

# Main chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Quick action buttons
st.markdown("""
    <div class="quick-actions">
        <div class="action-button">🎯 Find Agent</div>
        <div class="action-button">🔄 Compare</div>
        <div class="action-button">📊 Analytics</div>
        <div class="action-button">❓ Help</div>
    </div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.chat_history:
    if message.startswith("User:"):
        st.markdown(
            f'<div class="message"><div class="user-message">{message[6:]}</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="message"><div class="agent-message">{message[7:]}</div></div>',
            unsafe_allow_html=True
        )

# Chat input area
col1, col2 = st.columns([5,1])
with col1:
    user_input = st.text_input(
        "",
        placeholder="Ask about AI agents, task recommendations, or analysis...",
        key="chat_input"
    )
with col2:
    send_button = st.button("Send 🚀")

st.markdown("</div>", unsafe_allow_html=True)  # Close chat container

# Settings/Preferences Panel (as expandable section)
with st.expander("🎮 Agent Preferences", expanded=False):
    st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
    cols = st.columns(3)
    
    with cols[0]:
        st.session_state.preferences['task_complexity'] = st.slider(
            "Task Complexity",
            1, 10, st.session_state.preferences['task_complexity'],
            help="How complex is your task?"
        )
    
    with cols[1]:
        st.session_state.preferences['autonomy_level'] = st.slider(
            "Autonomy Level",
            1, 10, st.session_state.preferences['autonomy_level'],
            help="How autonomous should the agent be?"
        )
    
    with cols[2]:
        st.session_state.preferences['task_category'] = st.selectbox(
            "Task Category",
            options=sorted(data['task_category'].unique()),
            index=0,
            help="What type of task are you working on?"
        )
    
    if st.button("🔍 Get Recommendations", use_container_width=True):
        query = st.session_state.preferences
        
        # Calculate similarity
        def similarity(row):
            score = 0
            try:
                score -= abs(int(row['task_complexity']) - int(query['task_complexity'])) * 2
                score -= abs(int(row['autonomy_level']) - int(query['autonomy_level'])) * 2
                score += 5 if str(row.get('task_category')) == str(query.get('task_category')) else 0
            except Exception:
                pass
            return score

        data['similarity'] = data.apply(similarity, axis=1)
        similar_agents = data.sort_values('similarity', ascending=False).head(3)

        # Format recommendations
        if not similar_agents.empty and similar_agents['similarity'].max() > -10:
            recommendation_text = "🎯 Here are your personalized agent recommendations:\n\n"
            for i, agent in similar_agents.iterrows():
                recommendation_text += f"#{i+1} {agent.get('agent_type', 'Unknown Agent')}\n"
                recommendation_text += f"• Task Category: {agent.get('task_category', 'N/A')}\n"
                recommendation_text += f"• Accuracy: {agent.get('accuracy_score', 0):.2f}%\n"
                recommendation_text += f"• Cost: ${agent.get('cost_per_task_cents', 0)/100:.2f} per task\n\n"
            
            st.session_state.chat_history.append(f"Agent: {recommendation_text}")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Process chat input
if user_input and (send_button or True):
    # Add user message
    st.session_state.chat_history.append(f"User: {user_input}")
    
    try:
        # Get agent response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI agent advisor helping users find and understand different AI agents."},
                {"role": "user", "content": user_input}
            ]
        )
        agent_response = response['choices'][0]['message']['content']
        
        # Add agent response
        st.session_state.chat_history.append(f"Agent: {agent_response}")
        
        # Update UI
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.chat_history.append(f"Agent: I encountered an error. Please try again.")