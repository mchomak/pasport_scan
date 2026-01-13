"""Telegram bot keyboards."""
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def get_export_keyboard() -> InlineKeyboardMarkup:
    """Get export format selection keyboard."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="CSV", callback_data="export:csv"),
            InlineKeyboardButton(text="Excel", callback_data="export:excel"),
        ]
    ])
    return keyboard
