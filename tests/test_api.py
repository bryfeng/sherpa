from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth import optional_auth
from app.types import ChatResponse

HEADERS = {"Content-Type": "application/json"}


@pytest.fixture
def client(monkeypatch):
    async def _fake_health_check(*_args, **_kwargs) -> Dict[str, Any]:
        return {"status": "healthy"}

    async def _fake_run_chat(request):
        return ChatResponse(reply="ok", panels={}, sources=[], conversation_id=request.conversation_id)

    monkeypatch.setattr("app.api.health.AlchemyProvider.health_check", _fake_health_check)
    monkeypatch.setattr("app.api.health.CoingeckoProvider.health_check", _fake_health_check)
    monkeypatch.setattr("app.api.chat.run_chat", _fake_run_chat)

    app.dependency_overrides[optional_auth] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides = {}


class TestAgenticWalletAPI:
    """Test suite for Agentic Wallet Chat API"""

    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "providers" in data
        assert "database" in data
        assert "executions" in data
        assert "uptime_seconds" in data
        assert "version" in data

        # Should have alchemy and coingecko providers
        assert "alchemy" in data["providers"]
        assert "coingecko" in data["providers"]

    def test_chat_basic_request(self, client):
        """Test basic chat functionality"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, can you help me?"
                }
            ],
            "chain": "ethereum"
        }
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "reply" in data
        assert "panels" in data
        assert "sources" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0

    def test_chat_portfolio_analysis(self, client):
        """Test chat with portfolio analysis"""
        test_address = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"What is in wallet {test_address}?"
                }
            ],
            "address": test_address,
            "chain": "ethereum"
        }
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "reply" in data
        assert "panels" in data
        assert "sources" in data
        
        # Should have portfolio data in panels
        if "portfolio" in data["panels"]:
            portfolio = data["panels"]["portfolio"]
            assert "address" in portfolio
            assert "chain" in portfolio
            assert "total_value_usd" in portfolio
            assert "token_count" in portfolio
            assert "tokens" in portfolio
            assert portfolio["address"].lower() == test_address.lower()

    def test_chat_conversation_history(self, client):
        """Test chat with conversation history"""
        payload = {
            "messages": [
                {
                    "role": "user", 
                    "content": "Hello"
                },
                {
                    "role": "assistant",
                    "content": "I specialize in wallet portfolio analysis."
                },
                {
                    "role": "user",
                    "content": "Can you analyze a wallet for me?"
                }
            ],
            "chain": "ethereum"
        }
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "reply" in data
        assert len(data["reply"]) > 0

    def test_chat_invalid_request(self, client):
        """Test chat with invalid request format"""
        payload = {"invalid": "request"}
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0
        assert "messages" in str(data["detail"])

    def test_chat_missing_messages(self, client):
        """Test chat request without required messages field"""
        payload = {
            "address": "0x742d35cc6af4152f02006ad148e58b9e9a3e9b38",
            "chain": "ethereum"
        }
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 422

    def test_chat_empty_messages(self, client):
        """Test chat with empty messages array"""
        payload = {
            "messages": [],
            "chain": "ethereum"
        }
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200  # Should handle gracefully

    def test_chat_different_chains(self, client):
        """Test chat with different blockchain networks"""
        chains = ["ethereum", "polygon", "arbitrum"]
        
        for chain in chains:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "chain": chain
            }
            
            response = client.post("/chat", headers=HEADERS, json=payload)
            assert response.status_code == 200

    def test_chat_response_structure(self, client):
        """Test that chat response always has expected structure"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test message"
                }
            ]
        }
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200
        
        data = response.json()
        
        # Required fields
        assert "reply" in data
        assert "panels" in data  
        assert "sources" in data
        
        # Correct types
        assert isinstance(data["reply"], str)
        assert isinstance(data["panels"], dict)
        assert isinstance(data["sources"], list)

    def test_chat_with_invalid_address(self, client):
        """Test chat with invalid wallet address"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "What is in wallet invalid-address?"
                }
            ],
            "address": "invalid-address",
            "chain": "ethereum"
        }
        
        response = client.post("/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200  # Should handle gracefully
        
        data = response.json()
        assert "reply" in data
        # Should provide a helpful error message in the reply
