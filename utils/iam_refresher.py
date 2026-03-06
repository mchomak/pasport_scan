"""Automatic Yandex Cloud IAM token refresher."""
import asyncio
import shutil
from typing import Optional

import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

IAM_TOKEN_URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
REFRESH_INTERVAL = 12 * 60 * 60  # 12 hours in seconds


def _mask_token(token: str) -> str:
    """Show first 8 and last 4 chars of a token for logging."""
    if len(token) <= 16:
        return token[:4] + "***"
    return token[:8] + "***" + token[-4:]


async def _fetch_iam_via_oauth(oauth_token: str) -> Optional[str]:
    """Exchange OAuth token for IAM token via Yandex Cloud API."""
    logger.info(
        "IAM: exchanging OAuth token via API",
        oauth_token_preview=_mask_token(oauth_token),
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            IAM_TOKEN_URL,
            json={"yandexPassportOauthToken": oauth_token},
        )
        if resp.status_code != 200:
            body = resp.text[:500]
            logger.error(
                "IAM: OAuth → IAM exchange failed",
                status_code=resp.status_code,
                response_body=body,
            )
            return None

        data = resp.json()
        token = data.get("iamToken")
        if token:
            logger.info(
                "IAM: token obtained via OAuth API",
                iam_token_preview=_mask_token(token),
            )
        return token


async def _fetch_iam_via_cli() -> Optional[str]:
    """Get IAM token using yc CLI (if available)."""
    yc_path = shutil.which("yc")
    if not yc_path:
        logger.info("IAM: yc CLI not found, skipping CLI method")
        return None

    logger.info("IAM: trying yc CLI", yc_path=yc_path)
    proc = await asyncio.create_subprocess_exec(
        yc_path, "iam", "create-token",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(
            "IAM: yc iam create-token failed",
            returncode=proc.returncode,
            stderr=stderr.decode().strip(),
        )
        return None

    token = stdout.decode().strip()
    if token:
        logger.info(
            "IAM: token obtained via yc CLI",
            iam_token_preview=_mask_token(token),
        )
    return token if token else None


async def refresh_iam_token() -> Optional[str]:
    """Refresh IAM token using OAuth token (preferred) or yc CLI fallback."""
    # Method 1: OAuth token → IAM via API
    if settings.yc_oauth_token:
        try:
            token = await _fetch_iam_via_oauth(settings.yc_oauth_token)
            if token:
                return token
            logger.warning("IAM: OAuth exchange returned no token")
        except Exception as e:
            logger.error("IAM: OAuth exchange exception", error=str(e))
    else:
        logger.info("IAM: YC_OAUTH_TOKEN not set, skipping OAuth method")

    # Method 2: yc CLI
    try:
        token = await _fetch_iam_via_cli()
        if token:
            return token
    except Exception as e:
        logger.error("IAM: yc CLI exception", error=str(e))

    logger.warning("IAM: all refresh methods failed")
    return None


def _apply_token(token: str) -> None:
    """Update the IAM token in settings."""
    settings.yc_iam_token = token
    logger.info("IAM: token updated in settings", preview=_mask_token(token))


async def start_iam_refresh_loop() -> None:
    """Refresh IAM token every 12 hours.

    Runs as a background asyncio task. The initial refresh is done
    in main() before this loop starts, so we skip straight to waiting.
    """
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        try:
            logger.info("IAM: periodic refresh starting")
            token = await refresh_iam_token()
            if token:
                _apply_token(token)
            else:
                logger.warning("IAM: periodic refresh failed, keeping current token")
        except Exception as e:
            logger.error("IAM: periodic refresh loop error", error=str(e))
