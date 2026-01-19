"""Application configuration using pydantic-settings."""
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Telegram Bot
    bot_token: str = Field(..., description="Telegram bot token")
    admin_ids: str = Field(..., description="Comma-separated list of admin Telegram user IDs")

    # Database
    database_url: str = Field(
        ...,
        description="Database connection URL (postgresql+asyncpg://...)"
    )

    # OCR Provider
    ocr_provider_model: Literal["yandex"] = Field(
        default="yandex",
        description="OCR provider to use"
    )

    # Yandex Cloud OCR
    yc_folder_id: str = Field(default="", description="Yandex Cloud folder ID")
    yc_auth_mode: Literal["iam_token", "api_key"] = Field(
        default="api_key",
        description="Yandex Cloud authentication mode"
    )
    yc_iam_token: str = Field(default="", description="Yandex Cloud IAM token")
    yc_api_key: str = Field(default="", description="Yandex Cloud API key")
    yc_ocr_endpoint: str = Field(
        default="https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText",
        description="Yandex OCR API endpoint"
    )
    ocr_document_model: str = Field(
        default="passport",
        description="OCR document model"
    )
    ocr_language_codes: str = Field(
        default="*",
        description="OCR language codes"
    )

    # OCR Limits
    ocr_max_file_mb: int = Field(
        default=10,
        description="Maximum file size in MB for OCR"
    )
    ocr_max_megapixels: int = Field(
        default=20,
        description="Maximum image size in megapixels"
    )
    ocr_rate_limit_rps: float = Field(
        default=1.0,
        description="OCR rate limit in requests per second"
    )

    # PDF Processing
    pdf_render_dpi: int = Field(
        default=200,
        description="DPI for rendering PDF pages to images"
    )

    # Storage
    tmp_dir: str = Field(
        default="./tmp",
        description="Temporary directory for file processing"
    )
    store_source_files: bool = Field(
        default=False,
        description="Whether to store source files"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    @field_validator("admin_ids")
    @classmethod
    def parse_admin_ids(cls, v: str) -> str:
        """Validate admin_ids format."""
        if not v:
            raise ValueError("admin_ids cannot be empty")
        # Validate that all parts are integers
        try:
            [int(x.strip()) for x in v.split(",")]
        except ValueError:
            raise ValueError("admin_ids must be comma-separated integers")
        return v

    def get_admin_ids(self) -> list[int]:
        """Get admin IDs as list of integers."""
        return [int(x.strip()) for x in self.admin_ids.split(",")]

    @property
    def ocr_max_file_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.ocr_max_file_mb * 1024 * 1024


# Global settings instance
settings = Settings()
