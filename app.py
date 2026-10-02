import os
import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

load_dotenv()

st.set_page_config(
    page_title="Kenya Health Surveillance Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Kenya Health Surveillance Assistant")
st.markdown(
    "Ask questions about Kenya's health indicators, disease burden, "
    "and surveillance data. Every answer is grounded in source documents."
)
st.divider()

@st.cache_resource
def load_engine():
    from query_engine import setup_query_engine
    return setup_query_engine()

with st.spinner("Loading health knowledge base..."):
    query_engine = load_engine()

st.success("Knowledge base ready. Ask your first question below.")

st.markdown("**Suggested questions:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Malaria in children"):
        st.session_state.question = "What is the malaria prevalence among children under five in Kenya?"

with col2:
    if st.button("Maternal mortality"):
        st.session_state.question = "What is Kenya's maternal mortality ratio?"

with col3:
    if st.button("HIV coverage"):
        st.session_state.question = "What is the antiretroviral therapy coverage in Kenya?"

question = st.text_input(
    "Your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is the TB incidence rate in Kenya?"
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Searching knowledge base..."):
        response = query_engine.query(question)

    st.markdown("### Answer")
    st.markdown(str(response))

    st.markdown("### Sources")
    for i, node in enumerate(response.source_nodes):
        source_file = node.metadata.get("file_name", "Unknown source")
        score = round(node.score, 3) if node.score else "N/A"
        excerpt = node.text[:200].replace("\n", " ")
        with st.expander(f"Source {i+1}: {source_file} (relevance: {score})"):
            st.markdown(f"*\"{excerpt}...\"*")

st.divider()
st.caption("ZENIK.AI · Kenya Health Surveillance RAG System · Built with LlamaIndex and Pinecone")