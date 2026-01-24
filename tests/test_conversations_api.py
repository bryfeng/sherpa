import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from app.main import app


@dataclass
class StubConversationContext:
    conversation_id: str
    owner_address: Optional[str]
    title: Optional[str]
    archived: bool
    created_at: datetime
    last_activity: datetime
    total_tokens: int = 0
    message_count: int = 0
    messages: List[Any] = field(default_factory=list)
    convex_id: Optional[str] = None
    compressed_history: Optional[str] = None
    episodic_focus: Optional[Dict[str, Any]] = None
    portfolio_context: Optional[Dict[str, Any]] = None


class StubContextManager:
    def __init__(self):
        self._conversations: Dict[str, StubConversationContext] = {}
        self._by_address: Dict[str, List[str]] = {}
        self.convex_client = None

    def list_conversations(self, address: str, limit: int = 20) -> List[Dict[str, Any]]:
        addr = address.lower()
        items: List[Dict[str, Any]] = []
        for conv_id in self._by_address.get(addr, [])[: max(1, limit)]:
            ctx = self._conversations.get(conv_id)
            if not ctx:
                continue
            items.append(
                {
                    "conversation_id": conv_id,
                    "title": ctx.title or None,
                    "last_activity": ctx.last_activity.isoformat(),
                    "message_count": ctx.message_count,
                    "archived": ctx.archived,
                }
            )
        return items

    def create_conversation_id(self, address: str) -> str:
        addr = address.lower()
        conv_id = f"{addr}-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        ctx = StubConversationContext(
            conversation_id=conv_id,
            owner_address=addr,
            title=None,
            archived=False,
            created_at=now,
            last_activity=now,
        )
        self._conversations[conv_id] = ctx
        self._by_address.setdefault(addr, []).insert(0, conv_id)
        return conv_id

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        archived: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        ctx = self._conversations.get(conversation_id)
        if not ctx:
            return None
        if title is not None:
            ctx.title = title
        if archived is not None:
            ctx.archived = bool(archived)
        ctx.last_activity = datetime.now()
        return {
            "conversation_id": conversation_id,
            "title": ctx.title or None,
            "last_activity": ctx.last_activity.isoformat(),
            "message_count": ctx.message_count,
            "archived": ctx.archived,
        }


class StubAgent:
    def __init__(self, context_manager: StubContextManager):
        self.context_manager = context_manager


@pytest.fixture
def client(monkeypatch):
    ctx = StubContextManager()
    monkeypatch.setattr("app.api.conversations._get_agent", lambda: StubAgent(ctx))

    with TestClient(app) as test_client:
        yield test_client


class TestConversationsAPI:
    """Minimal tests for conversations endpoints (requires server running)."""

    def test_create_list_update_conversation(self, client):
        addr = "0x1234567890abcdef1234567890abcdef12345678"

        # Create new conversation
        r = client.post("/conversations", json={"address": addr})
        assert r.status_code == 200
        data = r.json()
        assert "conversation_id" in data
        conv_id = data["conversation_id"]
        assert conv_id.lower().startswith(addr.lower())

        # List should include it
        r = client.get("/conversations", params={"address": addr})
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list)
        assert any(it.get("conversation_id") == conv_id for it in items)

        # Update title and archived
        new_title = "My Test Chat"
        r = client.patch(f"/conversations/{conv_id}", json={"title": new_title, "archived": True})
        assert r.status_code == 200
        upd = r.json()
        assert upd.get("title") == new_title
        assert upd.get("archived") is True
