# Kenya Health Surveillance Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that answers natural language questions about Kenya's health indicators, disease burden, and surveillance data. Every answer is grounded in source documents.

## Live Demo
https://health-rag-data-pipeline-gjjxxgchk3pn3rjshropqd.streamlit.app/

## What it does
- Accepts natural language health questions in a web interface
- Retrieves the most relevant chunks from a Kenya health knowledge base using semantic search
- Generates grounded, cited answers via GPT-4o
- Shows source citations and relevance scores for every answer

## Tech Stack
LlamaIndex · OpenAI GPT-4o · Pinecone · Streamlit · Python 3.11

## Architecture
Health Documents → Chunking (512 tokens) → OpenAI Embeddings → Pinecone Vector Store
User Question → Semantic Retrieval → GPT-4o Generation → Cited Answer

## Data
Kenya health surveillance data covering malaria, TB, HIV, maternal health, and non-communicable diseases.

## Portfolio Context
This project is part of a Computational Epidemiology and Health Data Science portfolio under ZENIK.AI — building production-grade AI systems for African health contexts.

GitHub: github.com/Mwauramos | Contact: mwauramos.n@gmail.com