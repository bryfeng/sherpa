from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import health, tools, chat, swap, conversations, entitlement, perps, llm, history_summary, auth, dca, news, webhooks, copy_trading, polymarket, session_wallet, smart_accounts, swig_wallets
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

# Add CORS middleware - configured for security
# In production, only allow specific origins
ALLOWED_ORIGINS = [
    "https://runsherpa.ai",
    "https://app.runsherpa.ai",
    "https://www.runsherpa.ai",
    # Development origins (remove in production if needed)
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
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
app.include_router(copy_trading.router, tags=["Copy Trading"])
app.include_router(polymarket.router, tags=["Polymarket"])
app.include_router(session_wallet.router, tags=["Session Wallet"])
app.include_router(smart_accounts.router, tags=["Smart Accounts"])
app.include_router(swig_wallets.router, tags=["Swig Wallets"])


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
async def _init_chain_registry() -> None:
    """Initialize the chain registry at startup.

    This fetches supported chains from Relay API and makes them available
    for bridge/swap operations. The registry is cached so this only makes
    one API call at startup.
    """
    import logging
    from .core.bridge.chain_registry import init_chain_registry

    logger = logging.getLogger(__name__)
    try:
        registry = await init_chain_registry()
        logger.info(
            "Chain registry initialized: %d chains available",
            registry.chain_count,
        )
    except Exception as e:
        logger.warning("Failed to initialize chain registry: %s", e)


@app.on_event("startup")
async def _start_runtime() -> None:
    if not settings.agent_runtime_enabled:
        return
    register_builtin_strategies()
    await get_runtime().ensure_started()


@app.on_event("startup")
async def _start_copy_trading_bridge() -> None:
    """Start the copy trading event bridge."""
    if getattr(settings, "copy_trading_enabled", True):
        try:
            from .core.copy_trading import start_copy_trading_bridge
            await start_copy_trading_bridge()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to start copy trading bridge: {e}")


@app.on_event("shutdown")
async def _stop_runtime() -> None:
    runtime = get_runtime()
    if runtime.is_running:
        await runtime.stop()


@app.on_event("shutdown")
async def _stop_copy_trading_bridge() -> None:
    """Stop the copy trading event bridge."""
    try:
        from .core.copy_trading import stop_copy_trading_bridge
        await stop_copy_trading_bridge()
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
