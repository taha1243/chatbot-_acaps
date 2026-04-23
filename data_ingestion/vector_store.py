"""
Vector store interface for Atlas-RAG.
Handles PostgreSQL + pgvector operations: table management, upsert, search.
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

from .parser import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with score and payload."""

    id: str
    score: float
    text: str
    file_name: str
    header_path: str
    url_slug: str
    base_url: str
    metadata: Dict[str, Any]

    @property
    def full_url(self) -> str:
        """Get the full URL for citation."""
        return self.url_slug or self.base_url


class VectorStore:
    """
    Vector store interface for PostgreSQL + pgvector.
    Handles all vector database operations.
    """

    STANDARD_COLUMNS = {
        "id",
        "text",
        "file_name",
        "header_path",
        "url_slug",
        "base_url",
        "last_updated",
        "chunk_index",
        "metadata",
    }

    def __init__(
        self,
        dsn: str = "postgresql://atlas:atlas@localhost:5432/atlas_rag",
        table_name: str = "atlas_knowledge",
        embedding_dimension: int = 1024,
    ):
        self.dsn = dsn
        self.table_name = table_name
        self.collection_name = table_name
        self.embedding_dimension = embedding_dimension

        logger.info("Connecting to PostgreSQL vector store")

    def _connect(self):
        """Create a PostgreSQL connection."""
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _vector_literal(self, embedding: List[float]) -> str:
        """Serialize a vector for pgvector casts."""
        return "[" + ",".join(f"{value:.12g}" for value in embedding) + "]"

    def _row_to_result(self, row: Dict[str, Any]) -> SearchResult:
        """Convert a database row to a SearchResult."""
        return SearchResult(
            id=str(row["id"]),
            score=float(row["score"]),
            text=row["text"] or "",
            file_name=row["file_name"] or "",
            header_path=row["header_path"] or "",
            url_slug=row["url_slug"] or "",
            base_url=row["base_url"] or "",
            metadata=row.get("metadata") or {},
        )

    def _parse_timestamp(self, value: Optional[str]) -> Optional[datetime]:
        """Parse stored timestamps safely."""
        if not value:
            return None
        return datetime.fromisoformat(value)

    def _table_exists(self, conn) -> bool:
        """Check whether the configured table exists."""
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) AS table_name", [self.table_name])
            row = cur.fetchone()
            return bool(row and row["table_name"])

    def create_collection(self, recreate: bool = False) -> bool:
        """Create the vector table if it doesn't exist."""
        with self._connect() as conn:
            created = False
            existed = self._table_exists(conn)

            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

                if existed and recreate:
                    logger.warning("Dropping existing vector table: %s", self.table_name)
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}").format(
                            sql.Identifier(self.table_name)
                        )
                    )
                    existed = False

                if not existed:
                    logger.info("Creating vector table: %s", self.table_name)
                    cur.execute(
                        sql.SQL(
                            """
                            CREATE TABLE IF NOT EXISTS {} (
                                id UUID PRIMARY KEY,
                                text TEXT NOT NULL,
                                file_name TEXT NOT NULL,
                                header_path TEXT NOT NULL,
                                url_slug TEXT NOT NULL DEFAULT '',
                                base_url TEXT NOT NULL DEFAULT '',
                                last_updated TIMESTAMPTZ,
                                chunk_index INTEGER NOT NULL,
                                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                                embedding vector({})
                            )
                            """
                        ).format(
                            sql.Identifier(self.table_name),
                            sql.SQL(str(self.embedding_dimension)),
                        )
                    )
                    cur.execute(
                        sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (file_name)").format(
                            sql.Identifier(f"{self.table_name}_file_name_idx"),
                            sql.Identifier(self.table_name),
                        )
                    )
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {} ON {} USING gin (header_path gin_trgm_ops)"
                        ).format(
                            sql.Identifier(f"{self.table_name}_header_path_trgm_idx"),
                            sql.Identifier(self.table_name),
                        )
                    )
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {} ON {} USING gin (text gin_trgm_ops)"
                        ).format(
                            sql.Identifier(f"{self.table_name}_text_trgm_idx"),
                            sql.Identifier(self.table_name),
                        )
                    )
                    created = True

            conn.commit()
            return created

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Insert document chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        insert_sql = sql.SQL(
            """
            INSERT INTO {} (
                id, text, file_name, header_path, url_slug, base_url,
                last_updated, chunk_index, metadata, embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s::vector
            )
            """
        ).format(sql.Identifier(self.table_name))

        total_upserted = 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_embeddings = embeddings[i:i + batch_size]
                    values = []

                    for chunk, embedding in zip(batch_chunks, batch_embeddings):
                        values.append(
                            (
                                str(uuid.uuid4()),
                                chunk.text,
                                chunk.file_name,
                                chunk.header_path,
                                chunk.url_slug,
                                chunk.base_url,
                                self._parse_timestamp(chunk.last_updated),
                                chunk.chunk_index,
                                Json(chunk.metadata),
                                self._vector_literal(embedding),
                            )
                        )

                    cur.executemany(insert_sql, values)
                    total_upserted += len(values)

            conn.commit()

        logger.info("Upserted %s rows to %s", total_upserted, self.table_name)
        return total_upserted

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        query_vector = self._vector_literal(query_embedding)
        params: List[Any] = [query_vector]
        conditions = []

        with self._connect() as conn:
            if not self._table_exists(conn):
                return []

            if filter_conditions:
                for field, value in filter_conditions.items():
                    if field in self.STANDARD_COLUMNS - {"metadata"}:
                        conditions.append(
                            sql.SQL("{} = %s").format(sql.Identifier(field))
                        )
                        params.append(value)
                    else:
                        conditions.append(sql.SQL("metadata ->> %s = %s"))
                        params.extend([field, str(value)])

            where_clause = sql.SQL("")
            if conditions:
                where_clause = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

            query = sql.SQL(
                """
                SELECT
                    id::text AS id,
                    1 - (embedding <=> %s::vector) AS score,
                    text,
                    file_name,
                    header_path,
                    url_slug,
                    base_url,
                    metadata
                FROM {}
                """
            ).format(sql.Identifier(self.table_name))
            query += where_clause
            query += sql.SQL(" ORDER BY embedding <=> %s::vector LIMIT %s")
            params.extend([query_vector, top_k])

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        results = [self._row_to_result(row) for row in rows]
        if score_threshold is not None:
            results = [row for row in results if row.score >= score_threshold]
        return results

    def search_by_keyword(
        self,
        keyword: str,
        field: str = "header_path",
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for documents containing a keyword in a specific field."""
        if field not in {"header_path", "text", "file_name"}:
            raise ValueError(f"Unsupported keyword search field: {field}")

        query = sql.SQL(
            """
            SELECT
                id::text AS id,
                1.0 AS score,
                text,
                file_name,
                header_path,
                url_slug,
                base_url,
                metadata
            FROM {}
            WHERE {} ILIKE %s
            ORDER BY chunk_index
            LIMIT %s
            """
        ).format(sql.Identifier(self.table_name), sql.Identifier(field))

        with self._connect() as conn:
            if not self._table_exists(conn):
                return []
            with conn.cursor() as cur:
                cur.execute(query, [f"%{keyword}%", top_k])
                rows = cur.fetchall()

        return [self._row_to_result(row) for row in rows]

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        keyword_boost: float = 0.3,
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword search."""
        import re

        semantic_results = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            score_threshold=None,
        )

        keyword_results: List[SearchResult] = []
        patterns = [
            r"article\s*(\d+)",
            r"section\s*(\d+)",
            r"chapitre\s*(\d+)",
            r"titre\s*(\d+)",
        ]

        query_lower = query_text.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if "article" in pattern:
                    keyword = f"Article {match.group(1)}"
                elif "section" in pattern:
                    keyword = f"Section {match.group(1)}"
                elif "chapitre" in pattern:
                    keyword = f"Chapitre {match.group(1)}"
                else:
                    keyword = f"Titre {match.group(1)}"

                kw_results = self.search_by_keyword(keyword, field="header_path", top_k=5)
                keyword_results.extend(kw_results)
                logger.info("Keyword search for '%s' found %s results", keyword, len(kw_results))

        result_map = {result.id: result for result in semantic_results}

        for result in keyword_results:
            if result.id in result_map:
                existing = result_map[result.id]
                existing.score = min(1.0, existing.score + keyword_boost)
            else:
                result.score = keyword_boost + 0.5
                result_map[result.id] = result

        all_results = sorted(result_map.values(), key=lambda item: item.score, reverse=True)
        filtered = [result for result in all_results if result.score >= score_threshold]
        return filtered[:top_k]

    def has_file(self, file_name: str) -> bool:
        """Check if a file exists in the vector store."""
        query = sql.SQL("SELECT COUNT(*) AS count FROM {} WHERE file_name = %s").format(
            sql.Identifier(self.table_name)
        )

        with self._connect() as conn:
            if not self._table_exists(conn):
                return False
            with conn.cursor() as cur:
                cur.execute(query, [file_name])
                row = cur.fetchone()
                return bool(row and row["count"] > 0)

    def get_file_last_updated(self, file_name: str) -> Optional[str]:
        """Get the last_updated timestamp for a file from the vector store."""
        query = sql.SQL(
            "SELECT MAX(last_updated) AS last_updated FROM {} WHERE file_name = %s"
        ).format(sql.Identifier(self.table_name))

        with self._connect() as conn:
            if not self._table_exists(conn):
                return None
            with conn.cursor() as cur:
                cur.execute(query, [file_name])
                row = cur.fetchone()

        last_updated = row["last_updated"] if row else None
        return last_updated.isoformat() if last_updated else None

    def delete_by_file(self, file_name: str) -> int:
        """Delete all chunks from a specific file."""
        query = sql.SQL("DELETE FROM {} WHERE file_name = %s").format(
            sql.Identifier(self.table_name)
        )

        with self._connect() as conn:
            if not self._table_exists(conn):
                return 0
            with conn.cursor() as cur:
                cur.execute(query, [file_name])
                deleted = cur.rowcount
            conn.commit()

        logger.info("Deleted rows from file: %s", file_name)
        return deleted

    def get_collection_info(self) -> Dict[str, Any]:
        """Get vector table information and stats."""
        try:
            with self._connect() as conn:
                if not self._table_exists(conn):
                    return {
                        "name": self.table_name,
                        "points_count": 0,
                        "status": "missing",
                        "config": {"dimension": self.embedding_dimension},
                    }

                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) AS points_count FROM {}").format(
                            sql.Identifier(self.table_name)
                        )
                    )
                    count_row = cur.fetchone()

                return {
                    "name": self.table_name,
                    "points_count": count_row["points_count"] if count_row else 0,
                    "status": "ok",
                    "config": {"dimension": self.embedding_dimension},
                }
        except Exception as exc:
            logger.error("Error getting vector table info: %s", exc)
            return {"error": str(exc)}

    def health_check(self) -> bool:
        """Check if PostgreSQL is healthy and pgvector is available."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            return True
        except Exception as exc:
            logger.error("PostgreSQL vector store health check failed: %s", exc)
            return False


class MockVectorStore:
    """
    Mock vector store for testing without PostgreSQL.
    Stores vectors in memory.
    """

    def __init__(self, table_name: str = "test_collection", embedding_dimension: int = 1024):
        self.table_name = table_name
        self.collection_name = table_name
        self.embedding_dimension = embedding_dimension
        self.points: Dict[str, Dict[str, Any]] = {}
        logger.info("MockVectorStore initialized: %s", table_name)

    def create_collection(self, recreate: bool = False) -> bool:
        if recreate:
            self.points = {}
        return True

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            self.points[point_id] = {
                "vector": embedding,
                "payload": chunk.to_dict(),
            }
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        import numpy as np

        results = []
        query_vec = np.array(query_embedding)

        for point_id, point in self.points.items():
            payload = point["payload"]

            if filter_conditions:
                skip = False
                for field, value in filter_conditions.items():
                    if payload.get(field) != value:
                        skip = True
                        break
                if skip:
                    continue

            vec = np.array(point["vector"])
            score = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))

            if score_threshold is not None and score < score_threshold:
                continue

            results.append(
                SearchResult(
                    id=point_id,
                    score=score,
                    text=payload.get("text", ""),
                    file_name=payload.get("file_name", ""),
                    header_path=payload.get("header_path", ""),
                    url_slug=payload.get("url_slug", ""),
                    base_url=payload.get("base_url", ""),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k
                        not in {
                            "text",
                            "file_name",
                            "header_path",
                            "url_slug",
                            "base_url",
                            "last_updated",
                            "chunk_index",
                        }
                    },
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def search_by_keyword(
        self,
        keyword: str,
        field: str = "header_path",
        top_k: int = 5,
    ) -> List[SearchResult]:
        keyword_lower = keyword.lower()
        results = []

        for point_id, point in self.points.items():
            payload = point["payload"]
            haystack = str(payload.get(field, "")).lower()
            if keyword_lower not in haystack:
                continue

            results.append(
                SearchResult(
                    id=point_id,
                    score=1.0,
                    text=payload.get("text", ""),
                    file_name=payload.get("file_name", ""),
                    header_path=payload.get("header_path", ""),
                    url_slug=payload.get("url_slug", ""),
                    base_url=payload.get("base_url", ""),
                    metadata={},
                )
            )

        return results[:top_k]

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        keyword_boost: float = 0.3,
    ) -> List[SearchResult]:
        import re

        semantic_results = self.search(query_embedding=query_embedding, top_k=top_k * 2)
        keyword_results = []

        match = re.search(r"article\s*(\d+)", query_text.lower())
        if match:
            keyword_results.extend(
                self.search_by_keyword(f"Article {match.group(1)}", field="header_path", top_k=5)
            )

        result_map = {result.id: result for result in semantic_results}
        for result in keyword_results:
            if result.id in result_map:
                result_map[result.id].score = min(1.0, result_map[result.id].score + keyword_boost)
            else:
                result.score = 0.5 + keyword_boost
                result_map[result.id] = result

        all_results = sorted(result_map.values(), key=lambda item: item.score, reverse=True)
        return [result for result in all_results if result.score >= score_threshold][:top_k]

    def has_file(self, file_name: str) -> bool:
        """Check if a file exists in the mock store."""
        return any(point["payload"].get("file_name") == file_name for point in self.points.values())

    def get_file_last_updated(self, file_name: str) -> Optional[str]:
        """Get timestamp from mock store."""
        for point in self.points.values():
            if point["payload"].get("file_name") == file_name:
                return point["payload"].get("last_updated")
        return None

    def delete_by_file(self, file_name: str) -> int:
        """Delete all chunks belonging to a file."""
        to_delete = [
            point_id
            for point_id, point in self.points.items()
            if point["payload"].get("file_name") == file_name
        ]
        for point_id in to_delete:
            del self.points[point_id]
        return len(to_delete)

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "name": self.table_name,
            "points_count": len(self.points),
            "status": "ok",
        }

    def health_check(self) -> bool:
        return True
