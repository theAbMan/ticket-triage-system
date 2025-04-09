from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_classify_ticket():
    response = client.post("/classify",json={"text":"My internet is down!"})
    assert response.status_code == 200
    assert "category" in response.json()