from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_analyze_opening():
    response = client.post(
        "/api/analyze_opening",
        json={"position": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"}
    )
    assert response.status_code == 200
    assert "opening" in response.json()
    assert "description" in response.json()

def test_greet():
    response = client.post(
        "/api/greet",
        json={"name": "Alice"}
    )
    assert response.status_code == 200
    assert "greeting" in response.json()
    assert "Alice" in response.json()["greeting"]