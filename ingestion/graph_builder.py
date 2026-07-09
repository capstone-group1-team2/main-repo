"""The ONLY module in this codebase that writes to Neo4j . 
Every write goes through parameterized Cypher via the driver —
nothing here is ever typed by hand into Neo4j Browser, and no other module
imports the neo4j driver.
"""

from __future__ import annotations

from neo4j import GraphDatabase

from ingestion.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from ingestion.graph_seed import CATEGORY_RELATIONS


class GraphBuilder:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USERNAME, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def init_schema(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE")

    def seed_category_relations(self):
        """Idempotent — MERGE means re-running this never creates duplicates."""
        with self.driver.session() as s:
            for a, b in CATEGORY_RELATIONS:
                s.run(
                    """
                    MERGE (a:Concept {name: $a})
                    MERGE (b:Concept {name: $b})
                    MERGE (a)-[:RELATED_TO]->(b)
                    """,
                    a=a,
                    b=b,
                )

    def link_chunk_to_category(self, chunk_id: str, category: str, source_file: str, version: int):
        """Called once per chunk actually added this run — fully data-driven."""
        with self.driver.session() as s:
            s.run(
                """
                MERGE (ch:Chunk {chunk_id: $chunk_id})
                SET ch.source_file = $source_file, ch.version = $version
                MERGE (c:Concept {name: $category})
                MERGE (c)-[:MENTIONED_IN]->(ch)
                """,
                chunk_id=chunk_id,
                category=category,
                source_file=source_file,
                version=version,
            )

    def deactivate_chunk(self, chunk_id: str):
        """Called for chunks hash_store.py flags as removed."""
        with self.driver.session() as s:
            s.run(
                "MATCH (ch:Chunk {chunk_id: $chunk_id}) DETACH DELETE ch",
                chunk_id=chunk_id,
            )

    def chunk_count(self) -> int:
        with self.driver.session() as s:
            return s.run("MATCH (ch:Chunk) RETURN count(ch) AS n").single()["n"]

    def concept_count(self) -> int:
        with self.driver.session() as s:
            return s.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"]

    def chunk_ids_present(self, chunk_ids) -> set:
        """Returns the subset of `chunk_ids` that actually exist in Neo4j
        right now. Used to self-heal after a downstream reset (e.g. the
        Neo4j volume was wiped) even though hash_store.json still thinks the
        source file is unchanged — see ARCHITECTURE.md §4.1's
        reproducible-from-scratch guarantee."""
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return set()
        with self.driver.session() as s:
            result = s.run(
                "MATCH (ch:Chunk) WHERE ch.chunk_id IN $ids RETURN ch.chunk_id AS chunk_id",
                ids=chunk_ids,
            )
            return {record["chunk_id"] for record in result}
