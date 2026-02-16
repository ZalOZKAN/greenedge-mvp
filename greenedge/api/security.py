"""API security middleware - authentication and rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

from greenedge.logging_config import get_logger
from greenedge.settings import settings

logger = get_logger("security")

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request) -> Optional[str]:
    """Verify API key if configured.
    
    If GREENEDGE_API_KEY is not set, authentication is disabled.
    """
    if not settings.api.api_key:
        return None  # Auth disabled
    
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        logger.warning(f"Missing API key from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )
    
    if api_key != settings.api.api_key:
        logger.warning(f"Invalid API key from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return api_key


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        self._requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this client."""
        now = time.time()
        window_start = now - self.window_size
        
        # Clean old requests
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if ts > window_start
        ]
        
        # Check limit
        if len(self._requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Record request
        self._requests[client_ip].append(now)
        return True
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for this client."""
        now = time.time()
        window_start = now - self.window_size
        current = len([ts for ts in self._requests[client_ip] if ts > window_start])
        return max(0, self.requests_per_minute - current)


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=settings.api.rate_limit_per_minute)


async def check_rate_limit(request: Request) -> None:
    """Check rate limit for incoming request."""
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": "60"},
        )


def create_security_middleware() -> Callable:
    """Create middleware that combines auth and rate limiting."""
    
    async def middleware(request: Request, call_next):
        # Skip for health endpoint
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check API key (if configured)
        await verify_api_key(request)
        
        # Check rate limit
        await check_rate_limit(request)
        
        # Add rate limit headers
        response = await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        response.headers["X-RateLimit-Remaining"] = str(rate_limiter.get_remaining(client_ip))
        response.headers["X-RateLimit-Limit"] = str(settings.api.rate_limit_per_minute)
        
        return response
    
    return middleware
