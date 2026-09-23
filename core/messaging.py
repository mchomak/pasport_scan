"""Messenger-neutral input contracts."""

from dataclasses import dataclass
from enum import Enum


class MessengerSource(str, Enum):
    """Supported messenger platforms."""

    TELEGRAM = "telegram"
    MAX = "max"


def _normalize_identifier(value: str | int | None) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass(frozen=True, slots=True)
class IncomingImage:
    """Image bytes and platform metadata shared by messenger adapters."""

    content: bytes
    filename: str | None
    source: MessengerSource
    external_user_id: str | None
    external_chat_id: str | None
    external_message_id: str | None
    external_username: str | None = None
    source_type: str = "photo"
    source_file_id: str | None = None
    source_page_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, (bytes, bytearray, memoryview)):
            raise TypeError("IncomingImage.content must be bytes-like")
        if not isinstance(self.content, bytes):
            object.__setattr__(self, "content", bytes(self.content))

        if not isinstance(self.source, MessengerSource):
            object.__setattr__(self, "source", MessengerSource(self.source))

        for field_name in (
            "external_user_id",
            "external_chat_id",
            "external_message_id",
            "external_username",
            "source_file_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_identifier(getattr(self, field_name)),
            )
