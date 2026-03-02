"""Automatic Yandex Cloud IAM token refresher."""
import asyncio
import subprocess
import shutil
from typing import Optional

import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

IAM_TOKEN_URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
REFRESH_INTERVAL = 12 * 60 * 60  # 12 hours in seconds


async def _fetch_iam_via_oauth(oauth_token: str) -> Optional[str]:
    """Exchange OAuth token for IAM token via Yandex Cloud API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            IAM_TOKEN_URL,
            json={"yandexPassportOauthToken": oauth_token},
        )
        resp.raise_for_status()
        return resp.json().get("iamToken")


async def _fetch_iam_via_cli() -> Optional[str]:
    """Get IAM token using yc CLI (if available)."""
    yc_path = shutil.which("yc")
    if not yc_path:
        return None

    proc = await asyncio.create_subprocess_exec(
        yc_path, "iam", "create-token",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.warning(
            "yc iam create-token failed",
            stderr=stderr.decode().strip(),
        )
        return None

    token = stdout.decode().strip()
    return token if token else None


async def refresh_iam_token() -> Optional[str]:
    """Refresh IAM token using OAuth token (preferred) or yc CLI fallback."""
    # Method 1: OAuth token → IAM via API
    if settings.yc_oauth_token:
        try:
            token = await _fetch_iam_via_oauth(settings.yc_oauth_token)
            if token:
                logger.info("IAM token refreshed via OAuth API")
                return token
        except Exception as e:
            logger.warning("OAuth → IAM exchange failed", error=str(e))

    # Method 2: yc CLI
    try:
        token = await _fetch_iam_via_cli()
        if token:
            logger.info("IAM token refreshed via yc CLI")
            return token
    except Exception as e:
        logger.warning("yc CLI token refresh failed", error=str(e))

    return None


def _apply_token(token: str) -> None:
    """Update the IAM token in settings."""
    settings.yc_iam_token = token


async def start_iam_refresh_loop() -> None:
    """Refresh IAM token every 12 hours.

    Runs as a background asyncio task. The initial refresh is done
    in main() before this loop starts, so we skip straight to waiting.
    """
    # Periodic refresh
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        try:
            token = await refresh_iam_token()
            if token:
                _apply_token(token)
            else:
                logger.warning("IAM token refresh returned nothing, keeping current token")
        except Exception as e:
            logger.error("IAM token refresh loop error", error=str(e))
