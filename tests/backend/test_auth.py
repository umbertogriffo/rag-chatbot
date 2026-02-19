def test_create_token(client):
    response = client.post("/api/auth/token", json={"username": "testuser"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_token_is_valid(client):
    # Get a token
    response = client.post("/api/auth/token", json={"username": "testuser"})
    token = response.json()["access_token"]

    # Use token to access protected endpoint
    response = client.get("/api/models/", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
