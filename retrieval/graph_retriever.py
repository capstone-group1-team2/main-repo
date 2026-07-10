"""Read-only Neo4j lookups over the concept graph ingestion built
(ARCHITECTURE.md §6-§7). This module NEVER writes to Neo4j — only
ingestion/graph_builder.py is allowed to (ARCHITECTURE.md §4.1).
"""

from __future__ import annotations

from neo4j import GraphDatabase

from app.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


class GraphRetriever:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USERNAME, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def is_reachable(self) -> bool:
        """Liveness check for GET /health (M5) — reuses this already-open
        driver instead of a throwaway per-request client."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def related_categories(self, categories) -> set:
        """Categories connected to any of `categories` via a RELATED_TO
        edge (either direction — CATEGORY_RELATIONS pairs are conceptually
        undirected even though stored as one directed edge per pair),
        excluding the input categories themselves."""
        categories = list(categories)
        if not categories:
            return set()
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (c:Concept)-[:RELATED_TO]-(related:Concept)
                WHERE c.name IN $categories AND NOT related.name IN $categories
                RETURN DISTINCT related.name AS name
                """,
                categories=categories,
            )
            return {record["name"] for record in result}
