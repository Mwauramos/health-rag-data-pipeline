# Kenya Health Surveillance RAG System

A deployed Retrieval Augmented Generation (RAG) system that answers natural language questions about Kenya health indicators and disease surveillance data, grounded in source documents with citations.

**Live Demo**: https://health-rag-data-pipeline-gjjxxgchk3pn3rjshropqd.streamlit.app

---

## What This Does

Querying Kenya health data typically requires knowing where to look — MOH reports, DHIS2 extracts, county bulletins. This system lets a user ask a question in plain English and get a cited answer drawn directly from the underlying documents, with relevance scores for each source.

Built to demonstrate production RAG architecture with evaluation: every answer is grounded, every source is cited, and the system is monitored through the companion LLMOps Pipeline project.

---

## Features

- Natural language querying of Kenya health surveillance documents
- GPT-4o for answer generation
- Pinecone vector store for semantic document retrieval
- Source citation and relevance scoring on every response
- Streamlit interface for interactive querying

---

## Tech Stack

- Python
- LlamaIndex (RAG framework)
- Pinecone (vector database)
- GPT-4o (OpenAI)
- Streamlit (interface)

---

## Project Structure

```
health-rag-data-pipeline/
├── ingestion/             # Document loading and embedding pipeline
├── retrieval/             # Query and retrieval logic
├── app.py                 # Streamlit interface
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/Mwauramos/health-rag-data-pipeline
cd health-rag-data-pipeline
pip install -r requirements.txt
# Add your OpenAI and Pinecone API keys to .env
streamlit run app.py
```

---

## Related Projects

This system is monitored by the [LLMOps Pipeline](https://github.com/Mwauramos/llmops-pipeline), which runs automated evaluation, tracks faithfulness and relevancy scores, and enforces deployment quality gates using MLflow.

---

## Context

Part of the ZENIK.AI portfolio — healthcare AI and data engineering projects for African health systems.

**Author**: Amos Mwaura  
**Contact**: mwauramos.n@gmail.com  
**Portfolio**: https://mwauramos.github.io/zenik-ai
