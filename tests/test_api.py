from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_query_similarity():
    response = client.post(
        "/query",
        json={
            "query": "What does FastAPI use for request validation?",
            "k": 2,
            "mode": "similarity"
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "similarity"
    assert body["total_hits"] <= 2
    assert "hits" in body

def test_query_similarity_with_scores():
    response = client.post(
        "/query",
        json={
            "query": "How do response models work?",
            "k": 2,
            "mode": "similarity_with_scores"
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "similarity_with_scores"
    assert body["hits"][0]["score"] is not None