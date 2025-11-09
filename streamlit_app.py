import openai
import streamlit as st
import pandas as pd
import os

# Set OpenAI API key from Streamlit secrets or environment variable
if 'openai' in st.secrets:
    openai.api_key = st.secrets['openai']['OPENAI_API_KEY']
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("OpenAI API key not found. Please set it in Streamlit Cloud secrets or as an environment variable.")
    st.markdown("""
    ### How to set up your API key:
    1. Get your OpenAI API key from: https://platform.openai.com/api-keys
    2. For Streamlit Cloud:
        - Go to your app settings
        - Click on 'Secrets'
        - Add your API key as:
        ```toml
        [openai]
        OPENAI_API_KEY = "your-api-key-here"
        ```
    3. For local development:
        - Set the environment variable: `export OPENAI_API_KEY="your-api-key-here"`
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

# Modern agentic design with custom CSS
st.markdown(
    """
    <style>
    body {background: #f7faff;}
    .main-title {color: #2b6cb0; font-size: 2.8em; font-weight: 700; text-align: center; margin-bottom: 30px;}
    .sidebar {background: #2b6cb0; color: #fff; border-radius: 16px; padding: 24px; margin-bottom: 24px;}
    .chat-bubble {background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #2b6cb033; padding: 24px; margin-bottom: 18px;}
    .recommendation {background: #e3f2fd; border-radius: 16px; padding: 24px; margin-top: 24px;}
    .agent-card {background: #f0f4f8; border-radius: 12px; padding: 18px; margin-bottom: 12px; border-left: 6px solid #2b6cb0;}
    .highlight {color: #2b6cb0; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown("<div class='main-title'>Agentic Task Gap Analysis</div>", unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.markdown("<div class='sidebar'><h3>Your Preferences</h3></div>", unsafe_allow_html=True)
    task_complexity = st.selectbox("Task Complexity", sorted(data['task_complexity'].unique()))
    autonomy_level = st.selectbox("Desired Autonomy Level", sorted(data['autonomy_level'].unique()))
    task_category = st.selectbox("Task Category", sorted(data['task_category'].unique()))

# Find most similar agents (not exact match)
query = {
    "task_complexity": task_complexity,
    "autonomy_level": autonomy_level,
    "task_category": task_category
}

# Compute similarity (numeric for complexity/autonomy, exact for category)
def similarity(row):
    score = 0
    score -= abs(int(row['task_complexity']) - int(query['task_complexity'])) * 2
    score -= abs(int(row['autonomy_level']) - int(query['autonomy_level'])) * 2
    score += 5 if row['task_category'] == query['task_category'] else 0
    return score

data['similarity'] = data.apply(similarity, axis=1)
similar_agents = data.sort_values('similarity', ascending=False).head(5)

if not similar_agents.empty and similar_agents['similarity'].max() > -10:
    st.markdown("<div class='recommendation'><h3>Most Relevant Agents & Research Directions</h3></div>", unsafe_allow_html=True)
    for _, agent in similar_agents.iterrows():
        st.markdown(
            f"<div class='agent-card'>"
            f"<span class='highlight'>{agent['agent_type']}</span> for <span class='highlight'>{agent['task_category']}</span> | "
            f"Model: <b>{agent['model_architecture']}</b> | Accuracy: <b>{agent['accuracy_score']:.2f}</b> | Cost: <b>${agent['cost_per_task_cents']:.4f}</b> | Human Intervention: <b>{'Yes' if agent['human_intervention_required'] else 'No'}</b>"
            f"</div>", unsafe_allow_html=True)

    # Research suggestions
    research_prompt = (
        f"Based on the following agent data: {similar_agents[['agent_type','task_category','model_architecture','accuracy_score','cost_per_task_cents','human_intervention_required']].to_dict(orient='records')}, "
        f"suggest research directions or projects to improve the performance of these agents."
    )
    try:
        research_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": research_prompt}]
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
else:
    st.markdown("<div class='recommendation'><h3>No Similar Agents Found</h3></div>", unsafe_allow_html=True)
    st.markdown("<div class='chat-bubble'>No agents closely match your preferences, but you can research new agent architectures or hybrid approaches for this gap.</div>", unsafe_allow_html=True)