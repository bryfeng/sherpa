"""
Webhook API Endpoints

Receive and process webhooks from blockchain data providers (Alchemy, Helius).
"""

from fastapi import APIRouter, HTTPException, Header, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import logging

from ..services.events import (
    EventMonitoringService,
    get_event_monitoring_service,
    ChainType,
    EventType,
    Subscription,
    WalletActivity,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks")


# =============================================================================
# Response Models
# =============================================================================


class SubscriptionRequest(BaseModel):
    """Request to subscribe to an address."""

    address: str
    chain: str  # "ethereum", "polygon", "solana", etc.
    label: Optional[str] = None
    event_types: Optional[List[str]] = None


class SubscriptionResponse(BaseModel):
    """Subscription response."""

    id: str
    address: str
    chain: str
    status: str
    label: Optional[str] = None
    webhook_id: Optional[str] = None
    created_at: int
    error_message: Optional[str] = None


class WebhookResponse(BaseModel):
    """Response after processing a webhook."""

    success: bool
    activities_processed: int = 0
    message: Optional[str] = None


# =============================================================================
# Webhook Endpoints (called by providers)
# =============================================================================


@router.post("/alchemy", response_model=WebhookResponse)
async def alchemy_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_alchemy_signature: Optional[str] = Header(None, alias="X-Alchemy-Signature"),
):
    """
    Receive Alchemy Address Activity webhooks.

    Called by Alchemy when monitored addresses have activity.
    """
    try:
        body = await request.body()

        service = get_event_monitoring_service()
        activities = await service.handle_webhook(
            provider="alchemy",
            payload=body,
            signature=x_alchemy_signature,
        )

        return WebhookResponse(
            success=True,
            activities_processed=len(activities),
        )

    except ValueError as e:
        logger.warning(f"Invalid Alchemy webhook: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing Alchemy webhook: {e}", exc_info=True)
        # Return 200 to avoid retries for processing errors
        return WebhookResponse(
            success=False,
            message=str(e),
        )


@router.post("/helius", response_model=WebhookResponse)
async def helius_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
):
    """
    Receive Helius webhooks for Solana.

    Called by Helius when monitored addresses have activity.
    """
    try:
        body = await request.body()

        # Helius may send signature in Authorization header
        signature = None
        if authorization and authorization.startswith("Bearer "):
            signature = authorization[7:]

        service = get_event_monitoring_service()
        activities = await service.handle_webhook(
            provider="helius",
            payload=body,
            signature=signature,
        )

        return WebhookResponse(
            success=True,
            activities_processed=len(activities),
        )

    except ValueError as e:
        logger.warning(f"Invalid Helius webhook: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing Helius webhook: {e}", exc_info=True)
        return WebhookResponse(
            success=False,
            message=str(e),
        )


# =============================================================================
# Subscription Management Endpoints
# =============================================================================


@router.post("/subscriptions", response_model=SubscriptionResponse)
async def create_subscription(
    request: SubscriptionRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    """
    Subscribe to events for an address.

    Creates a webhook subscription to monitor the address.
    """
    try:
        # Parse chain
        try:
            chain = ChainType(request.chain.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chain: {request.chain}. Valid chains: {[c.value for c in ChainType]}"
            )

        # Parse event types
        event_types = None
        if request.event_types:
            event_types = []
            for et in request.event_types:
                try:
                    event_types.append(EventType(et.lower()))
                except ValueError:
                    logger.warning(f"Unknown event type: {et}")

        service = get_event_monitoring_service()
        subscription = await service.subscribe_address(
            address=request.address,
            chain=chain,
            user_id=x_user_id,
            event_types=event_types,
            label=request.label,
        )

        return SubscriptionResponse(
            id=subscription.id,
            address=subscription.address,
            chain=subscription.chain.value,
            status=subscription.status.value,
            label=subscription.label,
            webhook_id=subscription.webhook_id,
            created_at=int(subscription.created_at.timestamp() * 1000),
            error_message=subscription.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating subscription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/subscriptions/{address}")
async def delete_subscription(
    address: str,
    chain: str,
):
    """
    Unsubscribe from events for an address.
    """
    try:
        chain_type = ChainType(chain.lower())

        service = get_event_monitoring_service()
        success = await service.unsubscribe_address(
            address=address,
            chain=chain_type,
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No subscription found for {address} on {chain}"
            )

        return {"success": True, "address": address, "chain": chain}

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid chain: {chain}")
    except Exception as e:
        logger.error(f"Error deleting subscription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscriptions", response_model=List[SubscriptionResponse])
async def list_subscriptions(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    chain: Optional[str] = None,
):
    """
    List all subscriptions.

    Optionally filter by user ID (from header) or chain.
    """
    try:
        service = get_event_monitoring_service()

        if x_user_id:
            subscriptions = await service.get_subscriptions_for_user(x_user_id)
        else:
            # Return all subscriptions (admin only in production)
            subscriptions = list(service._subscriptions.values())

        # Filter by chain if specified
        if chain:
            try:
                chain_type = ChainType(chain.lower())
                subscriptions = [s for s in subscriptions if s.chain == chain_type]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid chain: {chain}")

        return [
            SubscriptionResponse(
                id=sub.id,
                address=sub.address,
                chain=sub.chain.value,
                status=sub.status.value,
                label=sub.label,
                webhook_id=sub.webhook_id,
                created_at=int(sub.created_at.timestamp() * 1000),
                error_message=sub.error_message,
            )
            for sub in subscriptions
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing subscriptions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscriptions/{subscription_id}", response_model=SubscriptionResponse)
async def get_subscription(subscription_id: str):
    """Get a specific subscription by ID."""
    try:
        service = get_event_monitoring_service()
        subscription = await service.get_subscription(subscription_id)

        if not subscription:
            raise HTTPException(
                status_code=404,
                detail=f"Subscription {subscription_id} not found"
            )

        return SubscriptionResponse(
            id=subscription.id,
            address=subscription.address,
            chain=subscription.chain.value,
            status=subscription.status.value,
            label=subscription.label,
            webhook_id=subscription.webhook_id,
            created_at=int(subscription.created_at.timestamp() * 1000),
            error_message=subscription.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting subscription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Activity Query Endpoints
# =============================================================================


class ActivityResponse(BaseModel):
    """Wallet activity response."""

    id: str
    wallet_address: str
    chain: str
    event_type: str
    tx_hash: str
    block_number: int
    timestamp: int
    direction: str
    value_usd: Optional[float] = None
    counterparty_address: Optional[str] = None
    counterparty_label: Optional[str] = None
    is_copyable: bool = False


@router.get("/activity/{address}", response_model=List[ActivityResponse])
async def get_wallet_activity(
    address: str,
    chain: Optional[str] = None,
    limit: int = 50,
    event_type: Optional[str] = None,
):
    """
    Get recent activity for a wallet address.

    Query parameters:
    - chain: Filter by chain
    - limit: Max number of results (default 50)
    - event_type: Filter by event type (swap, transfer_in, etc.)
    """
    try:
        service = get_event_monitoring_service()

        if not service.convex_client:
            raise HTTPException(
                status_code=503,
                detail="Activity storage not configured"
            )

        params = {
            "address": address.lower(),
            "limit": min(limit, 100),
        }

        if chain:
            try:
                ChainType(chain.lower())
                params["chain"] = chain.lower()
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid chain: {chain}")

        if event_type:
            try:
                EventType(event_type.lower())
                params["eventType"] = event_type.lower()
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")

        activities = await service.convex_client.query(
            "walletActivity:getByAddress",
            params,
        )

        return [
            ActivityResponse(
                id=a["id"],
                wallet_address=a["walletAddress"],
                chain=a["chain"],
                event_type=a["eventType"],
                tx_hash=a["txHash"],
                block_number=a["blockNumber"],
                timestamp=a["timestamp"],
                direction=a["direction"],
                value_usd=a.get("valueUsd"),
                counterparty_address=a.get("counterpartyAddress"),
                counterparty_label=a.get("counterpartyLabel"),
                is_copyable=a.get("isCopyable", False),
            )
            for a in activities
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting wallet activity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
