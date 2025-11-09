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

# Streamlit app setup with modern design
st.set_page_config(
    page_title="Agentic Task Gap Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-modern design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    body {
        background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
        font-family: 'Inter', sans-serif;
        color: #1a1f36;
    }

    /* Modern Title */
    .main-title {
        background: linear-gradient(135deg, #2b6cb0 0%, #1e4e8c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8em;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Chat Container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    .chat-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.18);
    }

    /* Chat Header */
    .chat-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
    }
    .assistant-info {
        margin-left: 1rem;
    }
    .assistant-name {
        font-weight: 600;
        color: #2b6cb0;
    }
    .assistant-status {
        font-size: 0.8em;
        color: #64748b;
    }

    /* Chat Messages */
    .chat-message {
        display: flex;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-out;
    }
    .message-content {
        max-width: 80%;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        line-height: 1.5;
    }
    .user-message {
        justify-content: flex-end;
    }
    .user-message .message-content {
        background: linear-gradient(135deg, #2b6cb0 0%, #1e4e8c 100%);
        color: white;
        border-radius: 20px 20px 5px 20px;
    }
    .assistant-message {
        justify-content: flex-start;
    }
    .assistant-message .message-content {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #1a1f36;
        border-radius: 20px 20px 20px 5px;
    }

    /* Sidebar */
    .sidebar {
        background: linear-gradient(165deg, #2b6cb0 0%, #1e4e8c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }
    .sidebar h3 {
        font-size: 1.3em;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Input Area */
    .input-area {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 2rem;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 16px;
    }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2b6cb0;
        box-shadow: 0 0 0 2px rgba(43, 108, 176, 0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #1e4e8c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(43, 108, 176, 0.2);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes typing {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
    .typing-indicator {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
        background: #f1f5f9;
        border-radius: 16px;
        width: fit-content;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #2b6cb0;
        border-radius: 50%;
        animation: typing 1.5s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title with gradient effect and subtitle
st.markdown("""
    <div class='main-title'>
        <span style='font-size: 0.4em; display: block; margin-bottom: 0.5rem;'>AI-Powered</span>
        Agentic Task Gap Analysis
        <span style='font-size: 0.6em; display: block; margin-top: 0.5rem; opacity: 0.8;'>Intelligent Agent Recommendations</span>
    </div>
""", unsafe_allow_html=True)

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=50)
    st.markdown("<div class='sidebar'>", unsafe_allow_html=True)
    st.markdown("### Task Preferences")
    
    # Task complexity
    try:
        task_complexity = st.select_slider(
            "Task Complexity",
            options=sorted(data['task_complexity'].unique()),
            value=data['task_complexity'].median()
        )
    except Exception:
        task_complexity = st.slider("Task Complexity", 1, 10, 5)

    # Autonomy level
    try:
        autonomy_level = st.select_slider(
            "Desired Autonomy Level",
            options=sorted(data['autonomy_level'].unique()),
            value=data['autonomy_level'].median()
        )
    except Exception:
        autonomy_level = st.slider("Autonomy Level", 1, 10, 5)

    # Task category
    try:
        task_category = st.selectbox(
            "Task Category",
            options=sorted(data['task_category'].unique()),
            index=0
        )
    except Exception:
        task_category = st.text_input("Task Category", "general")

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat history
    st.markdown("<div class='sidebar' style='margin-top: 1rem;'>", unsafe_allow_html=True)
    st.markdown("### Chat History")
    for message in st.session_state.chat_history[-5:]:  # Show last 5 messages
        st.markdown(f"<div style='font-size: 0.9em; margin: 0.5rem 0;'>{message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Chat interface
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Chat header
st.markdown("""
    <div class='chat-header'>
        <img src='https://img.icons8.com/color/48/bot.png' style='width: 40px; height: 40px;'>
        <div class='assistant-info'>
            <div class='assistant-name'>AI Assistant</div>
            <div class='assistant-status'>Online • Ready to help</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.chat_history:
    if message.startswith("User:"):
        st.markdown(f"""
            <div class='chat-message user-message'>
                <div class='message-content'>{message[6:]}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='chat-message assistant-message'>
                <div class='message-content'>{message[7:]}</div>
            </div>
        """, unsafe_allow_html=True)

# Chat input
placeholders = [
    "Ask about agent recommendations...",
    "Describe your task requirements...",
    "Need help finding the right agent?",
    "What kind of agent are you looking for?",
]

col1, col2 = st.columns([6,1])
with col1:
    user_input = st.text_input(
        "Message the AI Assistant:",
        placeholder=random.choice(placeholders),
        key="chat_input"
    )
with col2:
    send_button = st.button("Send 📤")

if user_input and (send_button or True):  # Allow Enter key to also send
    # Add user message to chat
    st.session_state.chat_history.append(f"User: {user_input}")
    
    # Process user input
    try:
        # Show typing indicator
        st.session_state.is_typing = True
        st.rerun()

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant helping users find the right AI agents for their tasks."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        agent_response = response['choices'][0]['message']['content']
        
        # Add assistant response to chat
        st.session_state.chat_history.append(f"Agent: {agent_response}")
        
        # Reset typing indicator
        st.session_state.is_typing = False
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.chat_history.append(f"Agent: I apologize, but I encountered an error. Please try again.")

# Show typing indicator if assistant is typing
if st.session_state.is_typing:
    st.markdown("""
        <div class='typing-indicator'>
            <div class='typing-dot'></div>
            <div class='typing-dot'></div>
            <div class='typing-dot'></div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # Close chat container

# Button to get recommendations based on current preferences
if st.button("Get Recommendations for Current Preferences", key="get_recommendations"):
    # Similar agents calculation
    query = {
        "task_complexity": task_complexity,
        "autonomy_level": autonomy_level,
        "task_category": task_category
    }
    
    # Calculate similarity scores
    def similarity(row):
        score = 0
        try:
            score -= abs(int(row['task_complexity']) - int(query['task_complexity'])) * 2
        except Exception:
            pass
        try:
            score -= abs(int(row['autonomy_level']) - int(query['autonomy_level'])) * 2
        except Exception:
            pass
        score += 5 if str(row.get('task_category')) == str(query.get('task_category')) else 0
        return score

    data['similarity'] = data.apply(similarity, axis=1)
    similar_agents = data.sort_values('similarity', ascending=False).head(5)

    # Display recommendations
    if not similar_agents.empty and similar_agents['similarity'].max() > -10:
        recommendations = []
        for _, agent in similar_agents.iterrows():
            recommendations.append({
                'agent_type': agent.get('agent_type', 'N/A'),
                'task_category': agent.get('task_category', 'N/A'),
                'model_architecture': agent.get('model_architecture', 'N/A'),
                'accuracy_score': agent.get('accuracy_score', 0),
                'cost_per_task_cents': agent.get('cost_per_task_cents', 0),
                'human_intervention_required': agent.get('human_intervention_required', False)
            })
        
        # Format recommendations as a chat message
        recommendation_text = "Here are the top agent recommendations based on your preferences:\n\n"
        for i, rec in enumerate(recommendations, 1):
            recommendation_text += f"{i}. {rec['agent_type']} for {rec['task_category']}\n"
            recommendation_text += f"   • Model: {rec['model_architecture']}\n"
            recommendation_text += f"   • Accuracy: {rec['accuracy_score']:.2f}\n"
            recommendation_text += f"   • Cost: ${rec['cost_per_task_cents']:.4f} per task\n"
            recommendation_text += f"   • Human Intervention: {'Yes' if rec['human_intervention_required'] else 'No'}\n\n"
        
        st.session_state.chat_history.append(f"Agent: {recommendation_text}")
        st.rerun()