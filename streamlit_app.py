import openai
import streamlit as st
import pandas as pd
import os

# Try to get the OpenAI API key from different sources
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if we have an API key
if not openai.api_key:
    st.error("⚠️ OpenAI API key not found!")
    st.warning("Please set up your API key in the app secrets:")
    
    st.code("""
1. Go to https://streamlit.io/
2. Find this app: Agentic Task Gap Analysis
3. Click on "Manage app" ⚙️
4. Select "Secrets"
5. Add this exactly:

OPENAI_API_KEY = "your-api-key-here"

6. Replace "your-api-key-here" with your actual OpenAI API key
7. Click "Save"
8. Click "Reboot app"
    """)
    
    st.info("Need an OpenAI API key? Get one here: https://platform.openai.com/api-keys")
    st.stop()
    st.markdown("""
    ### How to set up your API key:
    1. Get your OpenAI API key from: https://platform.openai.com/api-keys
    2. In your Streamlit Cloud app:
        - Go to "Settings" ⚙️
        - Click on "Secrets"
        - Add your API key with name exactly as: `OPENAI_API_KEY`
        - Value should be your OpenAI API key
    3. Click "Save"
    4. Deploy your app again
    """)
    st.stop()

# Load the dataset
data_path = "agentic_ai_performance_dataset_20250622.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Dataset file not found: {data_path}")
    st.info("Please make sure the dataset file is in the same directory as the application.")
    st.stop()

# Streamlit app setup with modern design
st.set_page_config(page_title="Agentic Task Gap Analysis", layout="wide")

# Custom CSS for modern design
st.markdown(
    """
    <style>
    body {background: #f7faff; font-family: 'Arial', sans-serif;}
    .main-title {color: #2b6cb0; font-size: 2.5em; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .chat-container {background: #fff; border-radius: 16px; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1); padding: 20px; margin-bottom: 20px;}
    .chat-bubble {background: #e3f2fd; border-radius: 16px; padding: 15px; margin-bottom: 10px; font-size: 1em;}
    .user-bubble {background: #f0f4f8; border-radius: 16px; padding: 15px; margin-bottom: 10px; font-size: 1em; text-align: right;}
    .sidebar {background: #2b6cb0; color: white; padding: 20px; border-radius: 10px;}
    .button {background-color: #2b6cb0; color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer;}
    .button:hover {background-color: #1a4e8a;}
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown("<div class='main-title'>Agentic Task Gap Analysis</div>", unsafe_allow_html=True)

# Sidebar for navigation / preferences / history
with st.sidebar:
    st.markdown("<div class='sidebar'><h3>Your Preferences</h3></div>", unsafe_allow_html=True)
    # Preference selectors driven from the dataset
    try:
        task_complexity = st.selectbox("Task Complexity", options=sorted(data['task_complexity'].unique()))
    except Exception:
        # Fallback if column missing or not sortable
        task_complexity = st.text_input("Task Complexity", value="1")

    try:
        autonomy_level = st.selectbox("Desired Autonomy Level", options=sorted(data['autonomy_level'].unique()))
    except Exception:
        autonomy_level = st.text_input("Desired Autonomy Level", value="1")

    try:
        task_category = st.selectbox("Task Category", options=sorted(data['task_category'].unique()))
    except Exception:
        task_category = st.text_input("Task Category", value="general")

    st.markdown("<hr />", unsafe_allow_html=True)
    st.markdown("<div class='sidebar'><h3>Chat History</h3></div>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for chat in st.session_state.chat_history:
        st.write(chat)

# Chat interface
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
user_input = st.text_input("Ask the agent anything:", placeholder="E.g., What agent should I use for text processing?")
if user_input:
    st.session_state.chat_history.append(f"User: {user_input}")
    # Process user input
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )
        agent_response = response['choices'][0]['message']['content']
        st.session_state.chat_history.append(f"Agent: {agent_response}")
    except Exception as e:
        agent_response = f"Error: {str(e)}"
        st.session_state.chat_history.append(f"Agent: {agent_response}")
    st.write(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.write(f"<div class='chat-bubble'>{agent_response}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Recommendation flow: compute similar agents for current preferences
query = {
    "task_complexity": task_complexity,
    "autonomy_level": autonomy_level,
    "task_category": task_category
}

# Compute similarity (numeric for complexity/autonomy, exact for category)
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

def render_recommendations(similar_agents):
    if similar_agents.empty or similar_agents['similarity'].max() <= -10:
        st.markdown("<div class='recommendation'><h3>No Similar Agents Found</h3></div>", unsafe_allow_html=True)
        st.markdown("<div class='chat-bubble'>No agents closely match your preferences, but you can research new agent architectures or hybrid approaches for this gap.</div>", unsafe_allow_html=True)
        return None

    st.markdown("<div class='recommendation'><h3>Most Relevant Agents & Research Directions</h3></div>", unsafe_allow_html=True)
    summary_records = []
    for _, agent in similar_agents.iterrows():
        st.markdown(
            f"<div class='agent-card'>"
            f"<span class='highlight'>{agent.get('agent_type','N/A')}</span> for <span class='highlight'>{agent.get('task_category','N/A')}</span> | "
            f"Model: <b>{agent.get('model_architecture','N/A')}</b> | Accuracy: <b>{agent.get('accuracy_score',0):.2f}</b> | Cost: <b>${agent.get('cost_per_task_cents',0):.4f}</b> | Human Intervention: <b>{'Yes' if agent.get('human_intervention_required') else 'No'}</b>"
            f"</div>", unsafe_allow_html=True)
        summary_records.append({
            'agent_type': agent.get('agent_type'),
            'task_category': agent.get('task_category'),
            'model_architecture': agent.get('model_architecture'),
            'accuracy_score': agent.get('accuracy_score'),
            'cost_per_task_cents': agent.get('cost_per_task_cents'),
            'human_intervention_required': agent.get('human_intervention_required')
        })

    # Research suggestions via OpenAI
    research_prompt = (
        f"Based on the following agent data: {summary_records}, suggest research directions or projects to improve the performance of these agents."
    )
    try:
        research_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": research_prompt}],
            max_tokens=400
        )
        research_suggestion = research_response['choices'][0]['message']['content']
    except openai.error.AuthenticationError:
        st.error("OpenAI API key is invalid. Please check your API key in Streamlit Cloud secrets.")
        st.stop()
    except Exception as e:
        st.error(f"Error getting research suggestions: {str(e)}")
        research_suggestion = "Unable to generate research suggestions at this time."

    st.markdown("<div class='recommendation'><h3>Research Suggestions</h3></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-box'><p>{research_suggestion}</p></div>", unsafe_allow_html=True)
    return research_suggestion

# Allow the user to explicitly request recommendations (so chat and preferences are decoupled)
if st.button("Get recommendations for current preferences"):
    suggestion_text = render_recommendations(similar_agents)
    if suggestion_text:
        st.session_state.chat_history.append(f"Agent (recommendations): {suggestion_text}")