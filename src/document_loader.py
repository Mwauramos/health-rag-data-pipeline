# document_loader.py
# Loads and chunks health documents for the Kenya Surveillance RAG System
# ZENIK.AI — Computational Epidemiology Portfolio

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Load API keys from .env file
load_dotenv()

# ── What this does ──────────────────────────────────────────────────────────
# SimpleDirectoryReader reads every PDF in the docs folder
# SentenceSplitter cuts the text into chunks of 512 tokens
# with 50 token overlap so no sentence is lost at a boundary
# ────────────────────────────────────────────────────────────────────────────

def load_documents(docs_path: str = "docs") -> list:
    """Load all PDF documents from the docs folder."""
    print(f"Loading documents from: {docs_path}")
    
    reader = SimpleDirectoryReader(
        input_dir=docs_path,
        required_exts=[".pdf", ".txt"]
    )
    
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document pages")
    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    print("Chunking documents...")
    
    splitter = SentenceSplitter(
        chunk_size=512,      # tokens per chunk
        chunk_overlap=50     # overlap to avoid cutting sentences
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks")
    return nodes


def preview_chunks(nodes: list, num_chunks: int = 3):
    """Print a preview of the first few chunks."""
    print("\n── CHUNK PREVIEW ──────────────────────────────")
    for i, node in enumerate(nodes[:num_chunks]):
        print(f"\nChunk {i+1}:")
        print(node.text[:300])
        print("...")
    print("───────────────────────────────────────────────\n")


if __name__ == "__main__":
    # Run the full load and chunk pipeline
    documents = load_documents()
    nodes = chunk_documents(documents)
    preview_chunks(nodes)
    print("Day 2 complete — documents loaded and chunked successfully.")