# embedder.py
# Converts document chunks into embeddings and stores them in Pinecone
# ZENIK.AI — Computational Epidemiology Portfolio

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from document_loader import load_documents, chunk_documents

load_dotenv()

# ── What this does ──────────────────────────────────────────────────────────
# OpenAIEmbedding converts each chunk into 1536 numbers
# PineconeVectorStore stores those numbers in our kenya-surveillance index
# VectorStoreIndex ties everything together into a searchable index
# ────────────────────────────────────────────────────────────────────────────

def setup_embeddings():
    """Configure OpenAI embedding model."""
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    print("Embedding model ready: text-embedding-3-small")


def setup_pinecone():
    """Connect to Pinecone and return the vector store."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("kenya-surveillance")
    vector_store = PineconeVectorStore(pinecone_index=index)
    print("Pinecone vector store connected: kenya-surveillance")
    return vector_store


def embed_and_store(nodes: list, vector_store) -> VectorStoreIndex:
    """Embed chunks and store them in Pinecone."""
    print(f"Embedding and storing {len(nodes)} chunks...")
    print("This may take 1 to 2 minutes...")
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("All chunks embedded and stored in Pinecone.")
    return index


if __name__ == "__main__":
    # Full pipeline: load → chunk → embed → store
    setup_embeddings()
    vector_store = setup_pinecone()
    
    documents = load_documents()
    nodes = chunk_documents(documents)
    
    index = embed_and_store(nodes, vector_store)
    
    print("\nDay 3 complete — chunks embedded and stored in Pinecone.")
    print(f"Your kenya-surveillance index now contains searchable health data.")