from datetime import datetime
from fastapi import APIRouter

from ..providers.llm import get_available_providers, canonical_provider_name
from ..config import settings

router = APIRouter(prefix="/llm")


def _serialise_providers():
    providers_map = get_available_providers()
    providers_list = []
    for provider_id, info in providers_map.items():
        providers_list.append(
            {
                "id": provider_id,
                "display_name": info.get("display_name", provider_id.title()),
                "status": info.get("status", "unknown"),
                "default_model": info.get("default_model"),
                "description": info.get("description"),
                "reason": info.get("reason"),
                "models": info.get("models", []),
            }
        )
    return providers_list


@router.get("/providers")
async def list_llm_providers():
    providers = _serialise_providers()
    default_provider = canonical_provider_name(settings.llm_provider)

    # Determine a sane default model that exists in the returned set
    available_models = []
    for provider in providers:
        if provider.get("status") == "available":
            for model in provider.get("models", []):
                model_id = model.get("id")
                if model_id:
                    available_models.append(model_id)

    preferred_default_model = settings.resolve_default_model(default_provider)
    if preferred_default_model not in available_models and available_models:
        preferred_default_model = available_models[0]

    return {
        "providers": providers,
        "default_provider": default_provider,
        "default_model": preferred_default_model,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }
