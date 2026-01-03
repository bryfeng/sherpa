from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import health, tools, chat, swap, conversations, entitlement, perps, llm, history_summary, auth, dca, news, webhooks
from .api import relay as relay_api
from .agent_runtime.router import router as runtime_router
from .agent_runtime import get_runtime, register_builtin_strategies
from .middleware import RateLimitMiddleware
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

# Add rate limiting middleware (added after CORS so it runs first on requests)
if settings.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(tools.router, tags=["Tools"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(swap.router, tags=["Swap"])
app.include_router(relay_api.router, tags=["Relay"])
app.include_router(conversations.router, tags=["Conversations"])
app.include_router(entitlement.router, tags=["Entitlement"])
app.include_router(perps.router, tags=["Perps"])
app.include_router(llm.router, tags=["LLM"])
app.include_router(history_summary.router, tags=["History"])
app.include_router(runtime_router, tags=["Runtime"])
app.include_router(auth.router, tags=["Auth"])
app.include_router(dca.router, tags=["DCA Strategies"])
app.include_router(news.router, tags=["News"])
app.include_router(webhooks.router, tags=["Webhooks"])


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


@app.on_event("startup")
async def _start_runtime() -> None:
    if not settings.agent_runtime_enabled:
        return
    register_builtin_strategies()
    await get_runtime().ensure_started()


@app.on_event("shutdown")
async def _stop_runtime() -> None:
    runtime = get_runtime()
    if runtime.is_running:
        await runtime.stop()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
