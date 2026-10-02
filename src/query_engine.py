# query_engine.py
# Retrieves relevant chunks and generates grounded answers with citations
# ZENIK.AI — Computational Epidemiology Portfolio

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext

load_dotenv()

def setup_query_engine():
    """Connect to Pinecone index and set up the query engine."""
    
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    Settings.llm = OpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1
    )
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("kenya-surveillance")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=6,
        response_mode="tree_summarize"
    )
    
    print("Query engine ready.")
    return query_engine


def ask(query_engine, question: str):
    """Ask a health question and get a grounded answer with citations."""
    print(f"\nQuestion: {question}")
    print("Searching documents...")
    
    response = query_engine.query(question)
    
    print(f"\nAnswer:\n{response}")
    
    print("\nSources:")
    for i, node in enumerate(response.source_nodes):
        source_file = node.metadata.get("file_name", "Unknown source")
        score = round(node.score, 3) if node.score else "N/A"
        excerpt = node.text[:150].replace("\n", " ")
        print(f"  [{i+1}] {source_file} (relevance: {score})")
        print(f"      \"{excerpt}...\"")
    
    print("\n" + "="*60)
    return response


if __name__ == "__main__":
    query_engine = setup_query_engine()
    
    questions = [
        "What is the malaria prevalence among children under five in Kenya?",
        "What is Kenya's maternal mortality ratio?",
        "What percentage of adults in Kenya have hypertension?"
    ]
    
    for question in questions:
        ask(query_engine, question)
    
    print("\nDay 5 complete — citations are working.")