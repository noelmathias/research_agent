import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.configuration import (
    CollectionConfigurationInternal,
    ConfigurationParameter,
    HNSWConfigurationInternal,
)
from shared.config import get_settings
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import sqlite3

settings = get_settings()


class VectorStore:
    """
    ChromaDB wrapper.
    Handles collection management, upsert, and similarity search.
    Singleton — one client per process.
    """

    _instance: Optional["VectorStore"] = None
    _client: Optional[Any] = None

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            # Ensure persist directory exists before ChromaDB opens it
            persist_path = Path(settings.chroma_persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._repair_legacy_collection_configs(persist_path / "chroma.sqlite3")

            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _repair_legacy_collection_configs(self, sqlite_path: Path) -> None:
        """
        Chroma versions in this repo may have persisted collections with
        `config_json_str = '{}'`. Newer Chroma expects a typed configuration
        object containing `_type`, so we repair those rows in place.
        """
        if not sqlite_path.exists():
            return

        conn = sqlite3.connect(str(sqlite_path))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, config_json_str FROM collections"
            )
            rows = cur.fetchall()

            for collection_id, config_json_str in rows:
                normalized = (config_json_str or "").strip()
                if normalized and normalized != "{}":
                    continue

                legacy_params = self._load_legacy_hnsw_params(cur, collection_id)
                if legacy_params:
                    config = CollectionConfigurationInternal(
                        [
                            ConfigurationParameter(
                                name="hnsw_configuration",
                                value=HNSWConfigurationInternal.from_legacy_params(
                                    legacy_params
                                ),
                            )
                        ]
                    )
                else:
                    config = CollectionConfigurationInternal()

                cur.execute(
                    "UPDATE collections SET config_json_str = ? WHERE id = ?",
                    (config.to_json_str(), collection_id),
                )

            conn.commit()
        finally:
            conn.close()

    def _load_legacy_hnsw_params(
        self,
        cur: sqlite3.Cursor,
        collection_id: str,
    ) -> Dict[str, Any]:
        cur.execute(
            """
            SELECT key, str_value, int_value, float_value, bool_value
            FROM collection_metadata
            WHERE collection_id = ? AND key LIKE 'hnsw:%'
            """,
            (collection_id,),
        )
        rows = cur.fetchall()

        params: Dict[str, Any] = {}
        for key, str_value, int_value, float_value, bool_value in rows:
            if str_value is not None:
                params[key] = str_value
            elif int_value is not None:
                params[key] = int_value
            elif float_value is not None:
                params[key] = float_value
            elif bool_value is not None:
                params[key] = bool(bool_value)

        return params

    def get_or_create_collection(
        self, collection_name: Optional[str] = None
    ) -> chromadb.Collection:
        name = collection_name or settings.chroma_collection_name
        client = self._get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        collection = self.get_or_create_collection(collection_name)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()

        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return results

    def collection_count(self, collection_name: Optional[str] = None) -> int:
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        name = collection_name or settings.chroma_collection_name
        client = self._get_client()
        client.delete_collection(name=name)

    def list_collections(self) -> List[str]:
        client = self._get_client()
        return [c.name for c in client.list_collections()]
