import json


def test_create_session(client):
    response = client.post("/api/chat/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["title"] == "New Chat"


def test_list_sessions(client):
    # Create a session first
    client.post("/api/chat/sessions")
    response = client.get("/api/chat/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_chat_message(client):
    response = client.post(
        "/api/chat/",
        json={
            "message": "Hello, world!",
            "use_rag": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "session_id" in data


def test_get_session_messages(client):
    # Send a chat message first
    chat_response = client.post(
        "/api/chat/",
        json={"message": "Test message", "use_rag": False},
    )
    session_id = chat_response.json()["session_id"]

    # Get messages
    response = client.get(f"/api/chat/sessions/{session_id}/messages")
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) == 2  # user + assistant
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_delete_session(client):
    # Create a session
    create_response = client.post("/api/chat/sessions")
    session_id = create_response.json()["id"]

    # Delete it
    response = client.delete(f"/api/chat/sessions/{session_id}")
    assert response.status_code == 200

    # Verify it's gone
    response = client.get(f"/api/chat/sessions/{session_id}/messages")
    assert response.status_code == 404


def test_chat_websocket(client):
    with client.websocket_connect("/api/chat/ws/test-session") as websocket:
        websocket.send_text(json.dumps({"message": "Hello via WebSocket"}))

        # Collect all messages until "done"
        messages = []
        while True:
            data = websocket.receive_json()
            messages.append(data)
            if data.get("type") == "done":
                break

        # Should have token messages and a done message
        assert len(messages) >= 2
        assert messages[-1]["type"] == "done"
        token_messages = [m for m in messages if m["type"] == "token"]
        assert len(token_messages) > 0
