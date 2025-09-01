#!/bin/bash

echo "🚀 Setting up Agentic Wallet POC"
echo "================================"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Copy environment template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys:"
    echo "   - ALCHEMY_API_KEY: Get from https://alchemy.com"
    echo "   - COINGECKO_API_KEY: Optional, get from https://coingecko.com/api"
else
    echo ".env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Test the CLI: python cli.py portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
echo "3. Start the API: python main.py"
echo "4. Visit http://localhost:8000/docs for API documentation"
