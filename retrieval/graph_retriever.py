from __future__ import annotations
from neo4j import GraphDatabase
from app.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

class GraphRetriever:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USERNAME, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def is_reachable(self) -> bool:
        """Liveness check for the Neo4j connection."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def related_categories(self, categories: list[str] | set[str]) -> set[str]:
        """
        Retrieves categories connected via RELATED_TO edges.
        Uses execute_read to ensure read-only access and better error handling.
        """
        if not categories:
            return set()

        query = """
        MATCH (c:Concept)-[:RELATED_TO]-(related:Concept)
        WHERE c.name IN $names AND NOT related.name IN $names
        RETURN DISTINCT related.name AS name
        """

        # Internal helper to execute the query within a read transaction
        def _fetch_related(tx, names):
            result = tx.run(query, names=list(names))
            return {record["name"] for record in result}

        with self.driver.session() as session:
            return session.execute_read(_fetch_related, categories)