"""Main entry point for the Agentic Wallet API"""

if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
