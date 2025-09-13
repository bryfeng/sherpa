from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import health, tools, chat, swap
from .api import bungee as bungee_api
from .config import settings

# Create FastAPI app
app = FastAPI(
    title="Agentic Wallet API",
    description="Research-focused Web3 wallet explorer backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(tools.router, tags=["Tools"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(swap.router, tags=["Swap"])
app.include_router(bungee_api.router, tags=["Bungee"])


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "name": "Agentic Wallet API",
        "version": "0.1.0",
        "description": "Research-focused Web3 wallet explorer backend",
        "docs": "/docs",
        "health": "/healthz"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
