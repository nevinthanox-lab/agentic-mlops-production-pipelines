"""
STEP 5.2 - Embed runbooks and index into pgvector
Uses sentence-transformers for local embeddings (no API cost) and
LlamaIndex's PGVectorStore to persist into Postgres.
"""

import os
import torch  # noqa: F401  - MUST import before sentence-transformers/llama_index on Windows to avoid DLL init-order conflict
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

RUNBOOK_DIR = "knowledge_base/runbooks"

DB_USER = os.getenv("POSTGRES_USER", "sentinelops")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sentinelops_dev_pw")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "sentinelops")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, local


def main():
    print("Loading embedding model (first run downloads ~90MB, cached after)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    print(f"Reading runbooks from {RUNBOOK_DIR}...")
    documents = SimpleDirectoryReader(RUNBOOK_DIR).load_data()
    print(f"Loaded {len(documents)} runbook documents.")

    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name="sre_runbooks",
        embed_dim=384,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Embedding and indexing into pgvector (table: data_sre_runbooks)...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )

    print("Done. Runbooks embedded and stored in Postgres pgvector.")

    # Quick sanity check retrieval
    retriever = index.as_retriever(similarity_top_k=2)
    test_query = "latency is high but CPU looks completely normal"
    results = retriever.retrieve(test_query)
    print(f"\nSanity check query: '{test_query}'")
    for r in results:
        print(f"  -> {r.node.metadata.get('file_name')} (score={r.score:.3f})")


if __name__ == "__main__":
    main()
