import json
from datetime import datetime, timezone

from bot.conversation.chat_history import ChatHistory
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from backend.app.core.config import settings
from backend.app.core.security import get_current_user, verify_token
from backend.app.db.models import ChatMessage, ChatSession
from backend.app.db.session import get_session
from backend.app.llm_client import llm_client
from backend.app.schemas.chat import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
)

router = APIRouter()


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    statement = select(ChatSession).where(ChatSession.user_id == current_user).order_by(ChatSession.updated_at.desc())
    sessions = db.exec(statement).all()
    return sessions


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(
    model_name: str | None = None,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    session = ChatSession(
        user_id=current_user,
        model_name=model_name or settings.DEFAULT_MODEL,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_session_messages(
    session_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    # Verify session belongs to user
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != current_user:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Session not found")

    statement = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    messages = db.exec(statement).all()
    result = []
    for msg in messages:
        sources = json.loads(msg.sources) if msg.sources else None
        result.append(
            ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=sources,
                created_at=msg.created_at,
            )
        )
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != current_user:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Session not found")

    # Delete messages first
    statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
    messages = db.exec(statement).all()
    for msg in messages:
        db.delete(msg)

    db.delete(session)
    db.commit()
    return {"message": "Session deleted"}


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Non-streaming chat endpoint. For streaming, use the WebSocket endpoint."""

    # Get or create session
    session_id = request.session_id
    if session_id:
        chat_session = db.get(ChatSession, session_id)
        if not chat_session or chat_session.user_id != current_user:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Session not found")
    else:
        chat_session = ChatSession(
            user_id=current_user,
            model_name=request.model_name or settings.DEFAULT_MODEL,
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        session_id = chat_session.id

    # Save user message
    user_msg = ChatMessage(session_id=session_id, role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    # Build chat history from session messages
    statement = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    past_messages = db.exec(statement).all()
    chat_history = ChatHistory(total_length=10)
    for msg in past_messages:
        if msg.id != user_msg.id:
            chat_history.append(f"{msg.role}: {msg.content}")

    # Generate response - this is a simplified non-streaming version
    # In practice, the LLM client needs to be initialized with a loaded model
    # For now, return a placeholder that indicates the API is working

    full_response = llm_client.generate_answer(
        prompt=request.message,
        max_new_tokens=settings.DEFAULT_MAX_NEW_TOKENS,
    )
    response_text = (
        f"[API Connected] Received: '{request.message}' "
        f"(model: {request.model_name or settings.DEFAULT_MODEL}, "
        f"rag: {request.use_rag}, k: {request.k})"
        f"\n\n[LLM Response] {full_response}"
    )
    sources = None

    # Save assistant response
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response_text,
        sources=json.dumps(sources) if sources else None,
    )
    db.add(assistant_msg)

    # Update session timestamp
    chat_session.updated_at = datetime.now(timezone.utc)
    if chat_session.title == "New Chat":
        chat_session.title = request.message[:50]
    db.add(chat_session)
    db.commit()

    return ChatResponse(message=response_text, session_id=session_id, sources=sources)


@router.websocket("/ws/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming chat responses."""
    # Authenticate via query parameter or first message
    token = websocket.query_params.get("token")
    api_key = websocket.query_params.get("api_key")
    _user = "anonymous"

    if api_key and settings.API_KEYS and api_key in settings.API_KEYS:
        _user = "api_key_user"
    elif token:
        subject = verify_token(token)
        if subject:
            _user = subject

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            message = request.get("message", "")
            model_name = request.get("model_name", settings.DEFAULT_MODEL)
            use_rag = request.get("use_rag", True)
            k = request.get("k", settings.DEFAULT_K)

            # Send streaming response
            # In production, this would stream tokens from LamaCppClient
            response_text = f"[WS Connected] Received: '{message}' (model: {model_name}, rag: {use_rag}, k: {k})"

            # Stream word by word to simulate token streaming
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                await websocket.send_json({"type": "token", "content": chunk})

            await websocket.send_json({"type": "done", "content": response_text})

    except WebSocketDisconnect:
        pass
