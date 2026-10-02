"""Low-level Qdrant repository for collection lifecycle and vector point operations."""

import logging
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.app.vectorstore.config import VectorStoreConfig
from backend.app.vectorstore.models import VectorPoint, VectorStoreError, VectorStoreHealth

logger = logging.getLogger("app.vectorstore.repository")


class QdrantVectorRepository:
    """Repository managing Qdrant collection lifecycle, point upserts, lookups, and deletions."""

    def __init__(
        self,
        client: QdrantClient,
        config: Optional[VectorStoreConfig] = None,
    ):
        self.client = client
        self.config = config or VectorStoreConfig.from_settings()

    def _resolve_collection(self, collection_name: Optional[str] = None) -> str:
        return collection_name or self.config.collection_name

    def initialize_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> bool:
        """Idempotently initializes the Qdrant collection and verifies vector compatibility."""
        target_collection = self._resolve_collection(collection_name)
        target_dimension = dimension or self.config.dimension

        try:
            exists = self.client.collection_exists(collection_name=target_collection)
            if not exists:
                logger.info(
                    f"Creating Qdrant collection '{target_collection}' "
                    f"(dim={target_dimension}, distance=Cosine)"
                )
                self.client.create_collection(
                    collection_name=target_collection,
                    vectors_config=rest_models.VectorParams(
                        size=target_dimension,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
                return True

            # If collection exists, verify configuration compatibility
            info = self.client.get_collection(collection_name=target_collection)
            vectors_config = info.config.params.vectors

            # Handle both single VectorParams or Multi-vector configuration
            actual_size = None
            actual_distance = None
            if isinstance(vectors_config, rest_models.VectorParams):
                actual_size = vectors_config.size
                actual_distance = vectors_config.distance
            elif isinstance(vectors_config, dict) and "" in vectors_config:
                actual_size = vectors_config[""].size
                actual_distance = vectors_config[""].distance

            if actual_size is not None and actual_size != target_dimension:
                raise VectorStoreError(
                    f"Existing collection '{target_collection}' has incompatible dimension: "
                    f"expected {target_dimension}, found {actual_size}"
                )

            if actual_distance is not None and actual_distance != rest_models.Distance.COSINE:
                raise VectorStoreError(
                    f"Existing collection '{target_collection}' has incompatible distance: "
                    f"expected Cosine, found {actual_distance}"
                )

            logger.debug(f"Qdrant collection '{target_collection}' verified successfully.")
            return True

        except VectorStoreError:
            raise
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Failed to initialize collection '{target_collection}': {sanitized}")
            raise VectorStoreError(f"Collection initialization failed: {sanitized}") from e

    def upsert_points(
        self,
        points: List[VectorPoint],
        collection_name: Optional[str] = None,
    ) -> int:
        """Batch-upserts VectorPoint objects into the target collection with deterministic point IDs."""
        if not points:
            return 0

        target_collection = self._resolve_collection(collection_name)
        batch_size = self.config.batch_size
        total_upserted = 0

        try:
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                point_structs = [
                    rest_models.PointStruct(
                        id=p.id,
                        vector=p.vector,
                        payload=p.payload,
                    )
                    for p in batch
                ]

                self.client.upsert(
                    collection_name=target_collection,
                    points=point_structs,
                    wait=True,
                )
                total_upserted += len(batch)

            logger.info(
                f"Successfully upserted {total_upserted} vector points into '{target_collection}'"
            )
            return total_upserted

        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Failed to upsert points into '{target_collection}': {sanitized}")
            raise VectorStoreError(f"Vector upsert failed: {sanitized}") from e

    def get_point(
        self,
        point_id: str,
        collection_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single point by ID including its payload and vector."""
        target_collection = self._resolve_collection(collection_name)
        try:
            records = self.client.retrieve(
                collection_name=target_collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                return None
            record = records[0]
            return {
                "id": str(record.id),
                "vector": record.vector,
                "payload": record.payload or {},
            }
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Failed to retrieve point '{point_id}': {sanitized}")
            raise VectorStoreError(f"Point retrieval failed: {sanitized}") from e

    def get_document_point_ids(
        self,
        document_id: str,
        user_id: str,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Retrieves all existing Qdrant point IDs associated with a specific document and user."""
        target_collection = self._resolve_collection(collection_name)

        tenant_filter = rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="document_id",
                    match=rest_models.MatchValue(value=document_id),
                ),
                rest_models.FieldCondition(
                    key="user_id",
                    match=rest_models.MatchValue(value=user_id),
                ),
            ]
        )

        point_ids: List[str] = []
        offset = None

        try:
            while True:
                scroll_result, next_offset = self.client.scroll(
                    collection_name=target_collection,
                    scroll_filter=tenant_filter,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False,
                    offset=offset,
                )
                for record in scroll_result:
                    point_ids.append(str(record.id))

                if next_offset is None:
                    break
                offset = next_offset

            return point_ids
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(
                f"Failed to fetch point IDs for document '{document_id}': {sanitized}"
            )
            raise VectorStoreError(f"Failed to query document point IDs: {sanitized}") from e

    def search(
        self,
        query_vector: List[float],
        user_id: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Executes a tenant-isolated semantic vector search against Qdrant.
        
        Guarantees strict tenant isolation by applying a mandatory filter on user_id.
        Optionally scopes to a specific document_id.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required for tenant-isolated vector search.")
        if not query_vector:
            raise ValueError("query_vector cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        target_collection = self._resolve_collection(collection_name)

        must_conditions: List[rest_models.Condition] = [
            rest_models.FieldCondition(
                key="user_id",
                match=rest_models.MatchValue(value=user_id.strip()),
            )
        ]
        if document_id and document_id.strip():
            must_conditions.append(
                rest_models.FieldCondition(
                    key="document_id",
                    match=rest_models.MatchValue(value=document_id.strip()),
                )
            )

        query_filter = rest_models.Filter(must=must_conditions)

        try:
            # Modern qdrant-client uses query_points, legacy / mocks may use search
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=target_collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
                scored_points = response.points
            elif hasattr(self.client, "search"):
                scored_points = self.client.search(
                    collection_name=target_collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            else:
                raise VectorStoreError("Qdrant client does not support query_points or search API.")

            results: List[Dict[str, Any]] = []
            for p in scored_points:
                results.append({
                    "id": str(p.id),
                    "score": float(p.score),
                    "payload": p.payload or {},
                })
            return results

        except UnexpectedResponse as e:
            if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
                try:
                    self.initialize_collection(target_collection)
                    return []
                except Exception:
                    pass
            sanitized = str(e).replace("\\", "/")[:200]
            logger.error(f"Vector search failed in '{target_collection}' for user '{user_id}': {sanitized}")
            raise VectorStoreError(f"Vector search failed: {sanitized}") from e
        except Exception as e:
            sanitized = str(e).replace("\\", "/")[:200]
            logger.error(
                f"Vector search failed in '{target_collection}' for user '{user_id}': {sanitized}"
            )
            raise VectorStoreError(f"Vector search failed: {sanitized}") from e

    def delete_points(
        self,
        point_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> int:
        """Deletes a specific list of point IDs from the collection."""
        if not point_ids:
            return 0

        target_collection = self._resolve_collection(collection_name)
        try:
            self.client.delete(
                collection_name=target_collection,
                points_selector=rest_models.PointIdsList(points=point_ids),
                wait=True,
            )
            logger.info(
                f"Deleted {len(point_ids)} specific points from '{target_collection}'"
            )
            return len(point_ids)
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Failed to delete points from '{target_collection}': {sanitized}")
            raise VectorStoreError(f"Point deletion failed: {sanitized}") from e

    def delete_by_document(
        self,
        document_id: str,
        user_id: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Deletes all vector points for a document strictly isolated by tenant user_id."""
        target_collection = self._resolve_collection(collection_name)

        tenant_filter = rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="document_id",
                    match=rest_models.MatchValue(value=document_id),
                ),
                rest_models.FieldCondition(
                    key="user_id",
                    match=rest_models.MatchValue(value=user_id),
                ),
            ]
        )

        try:
            self.client.delete(
                collection_name=target_collection,
                points_selector=tenant_filter,
                wait=True,
            )
            logger.info(
                f"Deleted vectors for document '{document_id}' (user='{user_id}') from '{target_collection}'"
            )
            return 1
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(
                f"Failed to delete vectors for document '{document_id}': {sanitized}"
            )
            raise VectorStoreError(f"Vector deletion failed: {sanitized}") from e

    def count_points(
        self,
        collection_name: Optional[str] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> int:
        """Counts vector points in a collection with optional tenant and document scoping."""
        target_collection = self._resolve_collection(collection_name)
        conditions = []
        if user_id:
            conditions.append(
                rest_models.FieldCondition(
                    key="user_id",
                    match=rest_models.MatchValue(value=user_id),
                )
            )
        if document_id:
            conditions.append(
                rest_models.FieldCondition(
                    key="document_id",
                    match=rest_models.MatchValue(value=document_id),
                )
            )

        filter_obj = rest_models.Filter(must=conditions) if conditions else None

        try:
            res = self.client.count(
                collection_name=target_collection,
                count_filter=filter_obj,
                exact=True,
            )
            return res.count
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Failed to count points in '{target_collection}': {sanitized}")
            raise VectorStoreError(f"Point count failed: {sanitized}") from e

    def check_health(self, collection_name: Optional[str] = None) -> VectorStoreHealth:
        """Performs a non-destructive health and connectivity check on Qdrant."""
        target_collection = self._resolve_collection(collection_name)
        try:
            exists = self.client.collection_exists(collection_name=target_collection)
            points_cnt = 0
            dim = self.config.dimension
            distance_str = "Cosine"

            if exists:
                info = self.client.get_collection(collection_name=target_collection)
                points_cnt = info.points_count or 0
                vectors_cfg = info.config.params.vectors
                if isinstance(vectors_cfg, rest_models.VectorParams):
                    dim = vectors_cfg.size
                    distance_str = str(vectors_cfg.distance.name)

            return VectorStoreHealth(
                status="healthy",
                collection_name=target_collection,
                collection_exists=exists,
                points_count=points_cnt,
                dimension=dim,
                distance=distance_str,
            )
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            return VectorStoreHealth(
                status=f"unhealthy: {sanitized}",
                collection_name=target_collection,
                collection_exists=False,
                points_count=0,
                dimension=self.config.dimension,
                distance="Cosine",
            )
