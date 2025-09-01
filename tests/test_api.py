import requests
import json
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

class TestAgenticWalletAPI:
    """Test suite for Agentic Wallet Chat API"""

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = requests.get(f"{BASE_URL}/healthz")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "providers" in data
        assert "available_providers" in data
        assert "total_providers" in data
        
        # Should have alchemy and coingecko providers
        assert "alchemy" in data["providers"]
        assert "coingecko" in data["providers"]

    def test_chat_basic_request(self):
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
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "reply" in data
        assert "panels" in data
        assert "sources" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0

    def test_chat_portfolio_analysis(self):
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
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
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

    def test_chat_conversation_history(self):
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
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "reply" in data
        assert len(data["reply"]) > 0

    def test_chat_invalid_request(self):
        """Test chat with invalid request format"""
        payload = {"invalid": "request"}
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0
        assert "messages" in str(data["detail"])

    def test_chat_missing_messages(self):
        """Test chat request without required messages field"""
        payload = {
            "address": "0x742d35cc6af4152f02006ad148e58b9e9a3e9b38",
            "chain": "ethereum"
        }
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
        assert response.status_code == 422

    def test_chat_empty_messages(self):
        """Test chat with empty messages array"""
        payload = {
            "messages": [],
            "chain": "ethereum"
        }
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200  # Should handle gracefully

    def test_chat_different_chains(self):
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
            
            response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
            assert response.status_code == 200

    def test_chat_response_structure(self):
        """Test that chat response always has expected structure"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test message"
                }
            ]
        }
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
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

    def test_chat_with_invalid_address(self):
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
        
        response = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=payload)
        assert response.status_code == 200  # Should handle gracefully
        
        data = response.json()
        assert "reply" in data
        # Should provide a helpful error message in the reply

if __name__ == "__main__":
    # Simple test runner if pytest is not available
    import sys
    
    test_instance = TestAgenticWalletAPI()
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    print("Running Agentic Wallet API Tests...")
    print("=" * 50)
    
    for test_method in test_methods:
        try:
            method = getattr(test_instance, test_method)
            method()
            print(f"✅ {test_method}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method}: {str(e)}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
