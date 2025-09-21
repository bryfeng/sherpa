import requests

BASE_URL = "http://localhost:8000"


class TestConversationsAPI:
    """Minimal tests for conversations endpoints (requires server running)."""

    def test_create_list_update_conversation(self):
        addr = "0x1234567890abcdef1234567890abcdef12345678"

        # Create new conversation
        r = requests.post(f"{BASE_URL}/conversations", json={"address": addr})
        assert r.status_code == 200
        data = r.json()
        assert "conversation_id" in data
        conv_id = data["conversation_id"]
        assert conv_id.lower().startswith(addr.lower())

        # List should include it
        r = requests.get(f"{BASE_URL}/conversations", params={"address": addr})
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list)
        assert any(it.get("conversation_id") == conv_id for it in items)

        # Update title and archived
        new_title = "My Test Chat"
        r = requests.patch(f"{BASE_URL}/conversations/{conv_id}", json={"title": new_title, "archived": True})
        assert r.status_code == 200
        upd = r.json()
        assert upd.get("title") == new_title
        assert upd.get("archived") is True


if __name__ == "__main__":
    t = TestConversationsAPI()
    t.test_create_list_update_conversation()
    print("✅ Conversations API smoke test passed")

