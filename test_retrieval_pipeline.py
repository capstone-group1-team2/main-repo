import sys
import numpy as np
from retrieval.graph_retriever import GraphRetriever
from retrieval.weaviate_retriever import WeaviateRetriever
from retrieval.hybrid_retriever import HybridRetriever

# 1. Simple Mock Embedder to allow local testing without live API keys
class MockEmbedder:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_query(self, query: str) -> np.ndarray:
        print(f"   [Embedder] Generating mock embedding for query: '{query}'")
        # Generate a unit-normalized vector
        vec = np.random.randn(self.dimension)
        return vec / np.linalg.norm(vec)


def run_retrieval_diagnostics():
    print("=" * 60)
    print("     STARTING RAG RETRIEVAL PIPELINE DIAGNOSTICS")
    print("=" * 60)

    # Initialize dependencies
    embedder = MockEmbedder()
    graph_retriever = None
    weaviate_retriever = None
    hybrid_retriever = None

    try:
        # ----------------------------------------------------
        # STAGE 1: Test GraphRetriever (Neo4j Connection)
        # ----------------------------------------------------
        print("\n--- STAGE 1: Testing GraphRetriever ---")
        graph_retriever = GraphRetriever()
        
        if graph_retriever.is_reachable():
            print("Neo4j connection verified successfully.")
        else:
            print("Neo4j connection failed! Check your Neo4j container and .env credentials.")
            return

        # Test relation lookup
        test_categories = {"SHIPPING_POLICY"}
        print(f"-> Querying concepts related to categories: {test_categories}")
        related = graph_retriever.related_categories(test_categories)
        print(f"✅ Related concepts retrieved from Neo4j: {related}")

        # ----------------------------------------------------
        # STAGE 2: Test WeaviateRetriever (Vector Search)
        # ----------------------------------------------------
        print("\n--- STAGE 2: Testing WeaviateRetriever ---")
        weaviate_retriever = WeaviateRetriever(embedder=embedder)

        if weaviate_retriever.is_ready():
            print("Weaviate connection verified successfully.")
        else:
            print("Weaviate connection failed! Check your Weaviate container and port settings.")
            return

        # Test fetch all chunks (crucial for building the BM25 index later)
        print("-> Fetching all chunks currently stored in Weaviate...")
        all_chunks = weaviate_retriever.fetch_all_chunks()
        print(f"Successfully fetched {len(all_chunks)} chunks from Weaviate.")
        if len(all_chunks) == 0:
            print("WARNING: Weaviate collection is empty! Run ingestion first to seed data.")

        # Test semantic search
        test_query = "How do I cancel my order?"
        print(f"-> Executing vector search for: '{test_query}'")
        search_results = weaviate_retriever.search(test_query, top_k=2)
        print(f"Vector search returned {len(search_results)} results.")
        for idx, chunk in enumerate(search_results):
            print(f"   [{idx + 1}] ID: {chunk.chunk_id} | Score: {chunk.score:.4f} | Heading: {chunk.heading}")

        # ----------------------------------------------------
        # STAGE 3: Test HybridRetriever (Fusion & Graph Broadening)
        # ----------------------------------------------------
        print("\n--- STAGE 3: Testing HybridRetriever ---")
        print("-> Building in-memory BM25 index on Weaviate chunks...")
        hybrid_retriever = HybridRetriever(vec=weaviate_retriever, graph=graph_retriever)
        print("BM25 index built and HybridRetriever initialized successfully.")

        # Test standard retrieve (Dense + Sparse Fusion)
        print(f"-> Executing Hybrid Retrieval (RRF) for: '{test_query}'")
        hybrid_res = hybrid_retriever.retrieve(test_query)
        print("Hybrid Retrieval completed.")
        print(f"   - Confidence Level: {hybrid_res.confidence}")
        print(f"   - Top Dense Cosine Similarity: {hybrid_res.top_dense_score:.4f}")
        print(f"   - Max BM25 Score: {hybrid_res.bm25_max_score:.4f}")
        print(f"   - Number of Fused Chunks Returned: {len(hybrid_res.chunks)}")
        print(f"   - Contextually Related Concepts: {hybrid_res.related_concepts}")

        # Test Graph Broadening (Fallback mechanism when groundedness fails)
        if len(hybrid_res.chunks) > 0:
            print("\n-> Testing Graph Broadening query fallback...")
            broad_res = hybrid_retriever.broaden_via_graph(
                original_query=test_query, 
                first_pass_chunks=hybrid_res.chunks
            )
            print("Graph Broadening completed successfully.")
            print(f"   - Broadened Chunk Pool Size: {len(broad_res.chunks)}")
            print(f"   - Broadened Related Concepts: {broad_res.related_concepts}")

    except Exception as e:
        print(f"\nCRITICAL FAILURE during diagnostic run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Gracefully shut down active drivers
        print("\n--- Cleaning Up Connections ---")
        if hybrid_retriever:
            hybrid_retriever.close()
            print("HybridRetriever (and child connections) closed.")
        else:
            if weaviate_retriever:
                weaviate_retriever.close()
                print("Weaviate connection closed.")
            if graph_retriever:
                graph_retriever.close()
                print("Neo4j connection closed.")

    print("\n" + "=" * 60)
    print("     ALL RETRIEVAL TEST STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_retrieval_diagnostics()