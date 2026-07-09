"""The only module that talks to Weaviate for ingestion. Defines the chunk
collection schema and upserts/deletes objects by a content-addressed UUID
(uuid5 of chunk_id), so re-ingesting identical chunk text always maps to the
same object ID and is never duplicated.
"""

from __future__ import annotations

import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property

from ingestion.config import WEAVIATE_COLLECTION_NAME, WEAVIATE_GRPC_PORT, WEAVIATE_HOST, WEAVIATE_PORT

# Fixed namespace so chunk_id -> UUID mapping is stable across processes and runs.
_UUID_NAMESPACE = uuid.NAMESPACE_URL


def chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


class WeaviateLoader:
    def __init__(self):
        self._client = weaviate.connect_to_local(
            host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT
        )

    def close(self):
        self._client.close()

    def init_schema(self):
        if not self._client.collections.exists(WEAVIATE_COLLECTION_NAME):
            self._client.collections.create(
                WEAVIATE_COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="source_file", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="heading", data_type=DataType.TEXT),
                    Property(name="version", data_type=DataType.INT),
                    Property(name="text", data_type=DataType.TEXT),
                ],
            )

    def upsert_chunks(self, chunks: list, vectors, version: int):
        """`chunks` are always genuinely new content (hash_store only ever
        passes ADDED chunk_ids here), so a plain insert is correct — there is
        never an existing object at this UUID to overwrite."""
        collection = self._client.collections.get(WEAVIATE_COLLECTION_NAME)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(
                    uuid=chunk_uuid(chunk.chunk_id),
                    properties={
                        "chunk_id": chunk.chunk_id,
                        "source_file": chunk.source_file,
                        "category": chunk.category,
                        "heading": chunk.heading,
                        "version": version,
                        "text": chunk.text,
                    },
                    vector=[float(x) for x in vector],
                )
        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"Weaviate batch insert had {len(failed)} failure(s): {failed[:3]}")

    def delete_chunks(self, chunk_ids: list):
        if not chunk_ids:
            return
        collection = self._client.collections.get(WEAVIATE_COLLECTION_NAME)
        for chunk_id in chunk_ids:
            collection.data.delete_by_id(chunk_uuid(chunk_id))

    def count(self) -> int:
        collection = self._client.collections.get(WEAVIATE_COLLECTION_NAME)
        return collection.aggregate.over_all(total_count=True).total_count

    def chunk_ids_present(self, chunk_ids) -> set:
        """Returns the subset of `chunk_ids` that actually exist in Weaviate
        right now. Used to self-heal after a downstream reset (e.g. the
        Weaviate volume was wiped) even though hash_store.json still thinks
        the source file is unchanged — see ARCHITECTURE.md §4.1's
        reproducible-from-scratch guarantee."""
        collection = self._client.collections.get(WEAVIATE_COLLECTION_NAME)
        return {cid for cid in chunk_ids if collection.data.exists(chunk_uuid(cid))}
