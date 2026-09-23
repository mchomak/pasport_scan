"""Telegram bot handlers."""

from typing import Optional
import os
import sys

from aiogram import Bot, Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, BufferedInputFile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings
from core.messaging import IncomingImage, MessengerSource
from db.database import get_db
from db.repository import PassportRepository
from services.export_service import ExportService
from services.passport_processing import PassportProcessingService, PassportResult
from services.pdf_processor import PdfProcessor
from bot.keyboards import get_export_keyboard
from utils.logger import get_logger

logger = get_logger(__name__)
router = Router()

_processing_service: PassportProcessingService | None = None


def set_processing_service(service: PassportProcessingService) -> None:
    """Attach the service owned by the application runtime."""
    global _processing_service
    _processing_service = service


def _get_processing_service() -> PassportProcessingService:
    if _processing_service is None:
        raise RuntimeError("Telegram processing service is not configured")
    return _processing_service


# --- Command Handlers ---

@router.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    welcome_text = (
        "Добро пожаловать в Passport OCR Bot!\n\n"
        "Отправьте мне фото паспорта или PDF документ, "
        "и я распознаю данные паспорта.\n\n"
        "Поддерживаемые форматы:\n"
        "- Фотографии (JPEG, PNG)\n"
        "- PDF документы (многостраничные)\n\n"
        "Каждая страница обрабатывается отдельно."
    )
    await message.answer(welcome_text)


@router.message(Command("export"))
async def cmd_export(message: Message):
    """Handle /export command (admin only)."""
    admin_ids = settings.get_admin_ids()

    if message.from_user.id not in admin_ids:
        await message.answer("У вас нет доступа к этой команде.")
        return

    await message.answer(
        "Выберите формат выгрузки:",
        reply_markup=get_export_keyboard()
    )


# --- Callback Handlers ---

@router.callback_query(F.data.startswith("export:"))
async def handle_export_callback(callback: CallbackQuery):
    """Handle export format selection."""
    admin_ids = settings.get_admin_ids()

    if callback.from_user.id not in admin_ids:
        await callback.answer("У вас нет доступа к этой функции.", show_alert=True)
        return

    export_format = callback.data.split(":")[1]  # "csv" or "excel"

    await callback.message.edit_text("Формирую выгрузку...")

    try:
        async for session in get_db():
            repo = PassportRepository(session)
            records = await repo.get_all()
            record_count = len(records)

            logger.info(
                "Admin export requested",
                admin_id=callback.from_user.id,
                format=export_format,
                record_count=record_count
            )

            if record_count == 0:
                await callback.message.edit_text("Нет данных для выгрузки.")
                return

            if export_format == "csv":
                file_bytes = ExportService.export_csv(records)
                filename = "passports_export.csv"
                caption = f"Выгрузка {record_count} записей в формате CSV"
            else:
                file_bytes = ExportService.export_excel(records)
                filename = "passports_export.xlsx"
                caption = f"Выгрузка {record_count} записей в формате Excel"

            file = BufferedInputFile(file_bytes, filename=filename)
            await callback.message.answer_document(
                document=file,
                caption=caption
            )

            await callback.message.delete()
            break

    except Exception as e:
        logger.error("Export failed", error=str(e))
        try:
            await callback.message.edit_text(
                "Ошибка при формировании выгрузки."
            )
        except Exception:
            pass

    await callback.answer()


# --- Photo Handler ---

@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    """Handle photo messages."""
    logger.info(
        "Received photo",
        user_id=message.from_user.id,
        username=message.from_user.username
    )

    photo = message.photo[-1]

    if photo.file_size and photo.file_size > 20 * 1024 * 1024:
        await message.answer("Файл слишком большой. Максимальный размер: 20 МБ")
        return

    status_msg = await message.reply("Обрабатываю фото...")

    try:
        file = await bot.get_file(photo.file_id)
        file_bytes = await bot.download_file(file.file_path)
        image_bytes = file_bytes.read()

        await process_image(
            image_bytes=image_bytes,
            source_type="photo",
            source_file_id=photo.file_id,
            source_message_id=message.message_id,
            tg_user_id=message.from_user.id,
            tg_username=message.from_user.username,
            message=message,
            status_msg=status_msg
        )

    except Exception as e:
        logger.error("Photo processing failed", error=str(e))
        try:
            await status_msg.edit_text(
                "Произошла ошибка при обработке фото. "
                "Пожалуйста, попробуйте другое изображение."
            )
        except Exception:
            pass


# --- Document Handler ---

@router.message(F.document)
async def handle_document(message: Message, bot: Bot):
    """Handle document messages."""
    document = message.document

    logger.info(
        "Received document",
        user_id=message.from_user.id,
        username=message.from_user.username,
        mime_type=document.mime_type,
        file_name=document.file_name
    )

    if document.file_size and document.file_size > 20 * 1024 * 1024:
        await message.answer("Файл слишком большой. Максимальный размер: 20 МБ")
        return

    mime_type = document.mime_type or ""
    file_name = document.file_name or ""

    is_pdf = mime_type == "application/pdf" or file_name.lower().endswith(".pdf")
    is_image = (
        mime_type.startswith("image/") or
        file_name.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not (is_pdf or is_image):
        await message.answer(
            "Формат не поддерживается. "
            "Пожалуйста, отправьте изображение (JPEG, PNG) или PDF документ."
        )
        return

    status_msg = await message.reply("Обрабатываю документ...")

    try:
        file = await bot.get_file(document.file_id)
        file_bytes = await bot.download_file(file.file_path)
        content_bytes = file_bytes.read()

        if is_pdf:
            await process_pdf(
                pdf_bytes=content_bytes,
                source_file_id=document.file_id,
                source_message_id=message.message_id,
                tg_user_id=message.from_user.id,
                tg_username=message.from_user.username,
                message=message,
                status_msg=status_msg,
                filename=file_name or None,
            )
        else:
            await process_image(
                image_bytes=content_bytes,
                source_type="image_document",
                source_file_id=document.file_id,
                source_message_id=message.message_id,
                tg_user_id=message.from_user.id,
                tg_username=message.from_user.username,
                message=message,
                status_msg=status_msg,
                filename=file_name or None,
            )

    except Exception as e:
        logger.error("Document processing failed", error=str(e))
        try:
            await status_msg.edit_text(
                "Произошла ошибка при обработке документа. "
                "Пожалуйста, попробуйте другой файл."
            )
        except Exception:
            pass


# --- Processing Functions ---

async def _notify_rate_limit(status_msg: Message, wait_seconds: float) -> None:
    """Keep the existing Telegram wait notice while the shared service waits."""
    wait_display = max(int(wait_seconds), 1)
    try:
        await status_msg.edit_text(
            f"Минутный лимит запросов исчерпан. "
            f"Ожидайте ~{wait_display} сек., результат придёт автоматически."
        )
    except Exception:
        pass


def _telegram_incoming_image(
    image_bytes: bytes,
    source_type: str,
    source_file_id: str,
    source_message_id: int,
    tg_user_id: int,
    tg_username: Optional[str],
    message: Message,
    *,
    filename: Optional[str] = None,
    source_page_index: Optional[int] = None,
) -> IncomingImage:
    """Convert Telegram metadata to the shared messenger-neutral input."""
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    return IncomingImage(
        content=image_bytes,
        filename=filename,
        source=MessengerSource.TELEGRAM,
        external_user_id=tg_user_id,
        external_chat_id=chat_id,
        external_message_id=source_message_id,
        external_username=tg_username,
        source_type=source_type,
        source_file_id=source_file_id,
        source_page_index=source_page_index,
    )


def _format_telegram_result(
    result: PassportResult,
    response_prefix: str = "",
) -> str:
    """Wrap the shared result in the existing Telegram HTML presentation."""
    if not result.success:
        raise RuntimeError(result.error or "recognition failed")

    separator = "\n" if response_prefix else "\n\n"
    return (
        f"{response_prefix}<code>{result.format1}</code>"
        f"{separator}"
        f"<code>{result.format2}</code>\n"
        f"<blockquote expandable>{result.details}</blockquote>"
    )


async def _process_incoming_image(
    incoming: IncomingImage,
    status_msg: Message,
    *,
    response_prefix: str = "",
    error_text: str = (
        "Не удалось распознать паспорт. "
        "Пожалуйста, попробуйте более чёткий снимок."
    ),
) -> bool:
    """Process one common input and render it for Telegram."""
    try:
        result = await _get_processing_service().process_image(
            incoming,
            notify_wait=lambda seconds: _notify_rate_limit(status_msg, seconds),
        )
        await status_msg.edit_text(
            _format_telegram_result(result, response_prefix),
            parse_mode="HTML",
        )
        return True
    except Exception as e:
        logger.error("Image processing failed", error=str(e))
        try:
            await status_msg.edit_text(error_text)
        except Exception:
            pass
        return False


async def process_image(
    image_bytes: bytes,
    source_type: str,
    source_file_id: str,
    source_message_id: int,
    tg_user_id: int,
    tg_username: Optional[str],
    message: Message,
    status_msg: Message,
    *,
    filename: Optional[str] = None,
    source_page_index: Optional[int] = None,
    response_prefix: str = "",
):
    """Process one Telegram image through the shared processing service."""
    incoming = _telegram_incoming_image(
        image_bytes=image_bytes,
        source_type=source_type,
        source_file_id=source_file_id,
        source_message_id=source_message_id,
        tg_user_id=tg_user_id,
        tg_username=tg_username,
        message=message,
        filename=filename,
        source_page_index=source_page_index,
    )
    return await _process_incoming_image(
        incoming,
        status_msg,
        response_prefix=response_prefix,
    )


async def process_pdf(
    pdf_bytes: bytes,
    source_file_id: str,
    source_message_id: int,
    tg_user_id: int,
    tg_username: Optional[str],
    message: Message,
    status_msg: Message,
    *,
    filename: Optional[str] = None,
):
    """Extract PDF pages in Telegram and process them in the shared service."""
    try:
        pages = PdfProcessor.extract_pages_as_images(pdf_bytes)

        if not pages:
            await status_msg.edit_text("PDF не содержит страниц или не может быть обработан.")
            return

        await status_msg.edit_text(f"Обрабатываю PDF ({len(pages)} стр.)...")

        for image_bytes, page_index in pages:
            page_status = await message.reply(
                f"Обрабатываю страницу {page_index + 1}..."
            )
            incoming = _telegram_incoming_image(
                image_bytes=image_bytes,
                source_type="pdf_page",
                source_file_id=source_file_id,
                source_message_id=source_message_id,
                tg_user_id=tg_user_id,
                tg_username=tg_username,
                message=message,
                filename=filename,
                source_page_index=page_index,
            )
            await _process_incoming_image(
                incoming,
                page_status,
                response_prefix=f"Стр. {page_index + 1}\n",
                error_text=f"Страница {page_index + 1}: ошибка распознавания",
            )

        await status_msg.edit_text(f"PDF обработан ({len(pages)} стр.)")

    except Exception as e:
        logger.error("PDF processing failed", error=str(e))
        try:
            await status_msg.edit_text(
                "Не удалось обработать PDF. "
                "Пожалуйста, попробуйте другой файл."
            )
        except Exception:
            pass
