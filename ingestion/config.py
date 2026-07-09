"""Loads and validates every environment variable the ingestion pipeline needs.

This module and app/config.py are the ONLY two places in this codebase
allowed to read os.environ directly. Every
ingestion module (chunker, embedder, hash_store, weaviate_loader,
graph_builder, ingest) imports its settings from here.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill in a real value."
        )
    return value


# --- Paths -------------------------------------------------------------
CORPUS_DIR = os.environ.get("CORPUS_DIR", "data/corpus")
HASH_STORE_PATH = os.environ.get("HASH_STORE_PATH", "ingestion/hash_store.json")

# --- Embedding model -----------------------------------------------------
# Must match agent/groundedness.py's model exactly — query, document, and
# groundedness embeddings all need to live in the same vector space.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

# --- Semantic chunking (ingestion/chunker.py) -------------------------------
# Sentences on each side of a given sentence to combine into one group before
# embedding it for boundary detection (LlamaIndex SemanticSplitterNodeParser's
# "buffer size"). Percentile is the cosine-distance percentile between
# consecutive groups, within a heading section, above which a chunk boundary
# is inserted — higher means fewer, larger chunks.
CHUNK_BUFFER_SIZE = int(os.environ.get("CHUNK_BUFFER_SIZE", 1))
CHUNK_BREAKPOINT_PERCENTILE = float(os.environ.get("CHUNK_BREAKPOINT_PERCENTILE", 90))

# --- Weaviate --------------------------------------------------------------
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.environ.get("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", 50051))
WEAVIATE_COLLECTION_NAME = os.environ.get("WEAVIATE_COLLECTION_NAME", "Chunk")

# --- Neo4j -------------------------------------------------------------------
# No safe defaults for connection credentials — fail fast instead of silently
# trying to connect with the wrong password.
NEO4J_URI = _require("NEO4J_URI")
NEO4J_USERNAME = _require("NEO4J_USERNAME")
NEO4J_PASSWORD = _require("NEO4J_PASSWORD")
