import os
from pathlib import Path

os.environ["RAG_DB_PATH"] = "data/test-api-rag.db"

from fastapi.testclient import TestClient

from app.api import app, STORE


def setup_function() -> None:
    path = Path(STORE.path)
    if path.exists():
        path.unlink()
    # Recreate schema after deleting test database.
    from src.store import DDL
    with STORE.connect() as conn:
        conn.executescript(DDL)


def test_inline_ingest_and_query() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/documents/inline",
            json={
                "name": "policy.md",
                "text": "The service shall retain audit evidence for seven years. Administrative access requires authentication.",
            },
        )
        assert response.status_code == 201
        query = client.post(
            "/query",
            json={"question": "How long is audit evidence retained?", "k": 3, "prompt_mode": "zero_shot"},
        )
        assert query.status_code == 200
        payload = query.json()
        assert payload["citations"]
        assert payload["generator"] == "extractive"


def test_conflict_endpoint() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/requirements/conflicts",
            json={
                "minimum_overlap": 0.1,
                "requirements": [
                    {"requirement_id": "A", "source": "one", "text": "The API shall respond within 250 ms under peak load."},
                    {"requirement_id": "B", "source": "two", "text": "The API shall respond within 500 ms under peak load."},
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["conflict_count"] == 1
