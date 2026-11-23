"""The /v1 router shared by every gateway endpoint module (chat.py, usage.py).

Router-level dependencies: every route added here is authenticated, then rate-limited, by
default. enforce_rate_limit depends on require_api_key, and routes that need the client
declare it too; FastAPI resolves it once per request, so the key is looked up only once.
"""

from fastapi import APIRouter, Depends

from app.api.auth import require_api_key
from app.api.rate_limit import enforce_rate_limit

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
