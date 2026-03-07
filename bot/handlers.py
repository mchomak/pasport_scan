"""Telegram bot handlers."""
import io
from typing import Optional
from aiogram import Bot, Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, BufferedInputFile
from aiogram.fsm.context import FSMContext
from sqlalchemy.ext.asyncio import AsyncSession
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings
from db.database import get_db
from db.repository import PassportRepository
from ocr.provider import get_ocr_provider
from ocr.hybrid import HybridRecognizer
from ocr.openrouter import OpenRouterProvider
from services.image_processor import ImageProcessor
from services.pdf_processor import PdfProcessor
from services.export_service import ExportService
from bot.keyboards import get_export_keyboard
from utils.logger import get_logger
from utils.rate_limiter import MinuteRateLimiter
from utils.passport_formatter import (
    format_passport_type1,
    format_passport_type2,
    infer_gender,
)

logger = get_logger(__name__)
router = Router()

# Global OpenRouter rate limiter (shared across all users)
_openrouter_limiter = MinuteRateLimiter(rpm=settings.openrouter_rpm)


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
        # Get all records from database
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

            # Generate export file
            if export_format == "csv":
                file_bytes = ExportService.export_csv(records)
                filename = "passports_export.csv"
                caption = f"Выгрузка {record_count} записей в формате CSV"
            else:  # excel
                file_bytes = ExportService.export_excel(records)
                filename = "passports_export.xlsx"
                caption = f"Выгрузка {record_count} записей в формате Excel"

            # Send file
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

    # Get the largest photo
    photo = message.photo[-1]

    # Check file size
    if photo.file_size and photo.file_size > 20 * 1024 * 1024:
        await message.answer("Файл слишком большой. Максимальный размер: 20 МБ")
        return

    status_msg = await message.reply("Обрабатываю фото...")

    try:
        # Download photo
        file = await bot.get_file(photo.file_id)
        file_bytes = await bot.download_file(file.file_path)
        image_bytes = file_bytes.read()

        # Process
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

    # Check file size
    if document.file_size and document.file_size > 20 * 1024 * 1024:
        await message.answer("Файл слишком большой. Максимальный размер: 20 МБ")
        return

    # Determine file type
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
        # Download document
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
                status_msg=status_msg
            )
        else:  # is_image
            await process_image(
                image_bytes=content_bytes,
                source_type="image_document",
                source_file_id=document.file_id,
                source_message_id=message.message_id,
                tg_user_id=message.from_user.id,
                tg_username=message.from_user.username,
                message=message,
                status_msg=status_msg
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

async def _acquire_rate_limit(status_msg: Message) -> None:
    """Acquire an OpenRouter rate-limit slot; notify user if waiting."""
    if not _openrouter_limiter.is_enabled:
        return

    remaining = _openrouter_limiter.remaining()
    if remaining <= 0:
        wait_secs = _openrouter_limiter.seconds_until_free()
        wait_secs_display = max(int(wait_secs), 1)
        try:
            await status_msg.edit_text(
                f"Минутный лимит запросов исчерпан. "
                f"Ожидайте ~{wait_secs_display} сек., результат придёт автоматически."
            )
        except Exception:
            pass

    waited = await _openrouter_limiter.acquire()
    if waited > 0:
        logger.info("Rate limiter: waited %.1f s before processing", waited)


_PROVIDER_LABELS = {
    'rupasportread': 'Tesseract MRZ',
    'yandex_ocr': 'Yandex OCR',
    'openrouter': 'OpenRouter LLM',
    'inferred': 'Из имени',
    'none': '-',
}

_FIELD_LABELS = {
    'surname': 'Фамилия',
    'name': 'Имя',
    'middle_name': 'Отчество',
    'passport_number': 'Серия и номер',
    'birth_date': 'Дата рождения',
    'expiry_date': 'Срок действия',
    'gender': 'Пол',
    'birth_place': 'Место рождения',
}


def _format_details(
    passport_data,
    field_providers: dict,
    modules_used: list[str],
    per_module_data: dict,
) -> str:
    """Build collapsed detail text showing per-module results."""
    lines: list[str] = []

    # Per-module blocks
    priority = settings.get_module_priority()
    for module_key in priority:
        if module_key in per_module_data:
            label = _PROVIDER_LABELS.get(module_key, module_key)
            lines.append(f"[{label}]")
            data = per_module_data[module_key]
            for field_name, field_label in _FIELD_LABELS.items():
                val = getattr(data, field_name, None)
                if val is not None and str(val).strip():
                    lines.append(f"  + {field_label}: {val}")
                else:
                    lines.append(f"  - {field_label}: ---")
            lines.append("")

    # Skipped modules
    skipped = [
        _PROVIDER_LABELS.get(m, m) for m in priority
        if m not in per_module_data
    ]
    if skipped:
        lines.append(f"Пропущены: {', '.join(skipped)}")
        lines.append("")

    # Final merged result with source attribution
    lines.append("[Итог]")
    for field_name, field_label in _FIELD_LABELS.items():
        val = getattr(passport_data, field_name, None)
        if val is not None and str(val).strip():
            src = _PROVIDER_LABELS.get(field_providers.get(field_name, '?'), '?')
            lines.append(f"  {field_label}: {val}  ({src})")
        else:
            lines.append(f"  {field_label}: ---")

    return '\n'.join(lines)


async def process_image(
    image_bytes: bytes,
    source_type: str,
    source_file_id: str,
    source_message_id: int,
    tg_user_id: int,
    tg_username: Optional[str],
    message: Message,
    status_msg: Message
):
    """Process a single image through the hybrid OCR pipeline."""
    try:
        # Acquire rate-limit slot (may wait and notify user)
        if "openrouter" in settings.get_module_priority() and settings.openrouter_api_key:
            await _acquire_rate_limit(status_msg)

        # Normalize image
        processor = ImageProcessor()
        normalized_bytes, mime_type = processor.normalize_image(image_bytes)

        # Hybrid OCR recognition — only create providers that are in priority
        priority = settings.get_module_priority()
        yandex_provider = get_ocr_provider() if "yandex_ocr" in priority else None
        openrouter_provider = OpenRouterProvider() if "openrouter" in priority and settings.openrouter_api_key else None
        recognizer = HybridRecognizer(
            yandex_provider=yandex_provider,
            openrouter_provider=openrouter_provider,
        )
        hybrid_result = await recognizer.recognize(normalized_bytes, mime_type)

        passport_data = hybrid_result.passport_data
        modules_used = hybrid_result.modules_used

        # Infer gender from patronymic/surname if not detected
        if not passport_data.gender:
            inferred = infer_gender(passport_data.middle_name, passport_data.surname)
            if inferred:
                passport_data.gender = inferred
                hybrid_result.field_providers['gender'] = 'inferred'

        # Save to database
        async for session in get_db():
            repo = PassportRepository(session)

            quality_score = passport_data.count_filled_fields()

            record = await repo.create(
                tg_user_id=tg_user_id,
                tg_username=tg_username,
                source_type=source_type,
                source_file_id=source_file_id,
                source_message_id=source_message_id,
                source_page_index=None,
                passport_number=passport_data.passport_number,
                expiry_date=passport_data.expiry_date,
                surname=passport_data.surname,
                name=passport_data.name,
                middle_name=passport_data.middle_name,
                gender=passport_data.gender,
                birth_date=passport_data.birth_date,
                birth_place=passport_data.birth_place,
                raw_payload=hybrid_result.raw_response,
                quality_score=quality_score,
            )

            # Generate encoded formats
            format1 = format_passport_type1(passport_data)
            format2 = format_passport_type2(passport_data)

            # Build collapsed detail text
            details = _format_details(
                passport_data,
                hybrid_result.field_providers,
                modules_used,
                hybrid_result.per_module_data,
            )

            response_text = (
                f"<code>{format1}</code>\n"
                "\n"
                f"<code>{format2}</code>\n"
                f"<blockquote expandable>"
                f"{details}"
                f"</blockquote>"
            )

            await status_msg.edit_text(response_text, parse_mode="HTML")
            break

    except Exception as e:
        logger.error("Image processing failed", error=str(e))
        try:
            await status_msg.edit_text(
                "Не удалось распознать паспорт. "
                "Пожалуйста, попробуйте более чёткий снимок."
            )
        except Exception:
            pass


async def process_pdf(
    pdf_bytes: bytes,
    source_file_id: str,
    source_message_id: int,
    tg_user_id: int,
    tg_username: Optional[str],
    message: Message,
    status_msg: Message
):
    """Process PDF document page by page."""
    try:
        priority = settings.get_module_priority()

        # Extract pages as images
        pdf_processor = PdfProcessor()
        pages = pdf_processor.extract_pages_as_images(pdf_bytes)

        if not pages:
            await status_msg.edit_text("PDF не содержит страниц или не может быть обработан.")
            return

        await status_msg.edit_text(f"Обрабатываю PDF ({len(pages)} стр.)...")

        # Process each page
        for image_bytes, page_index in pages:
            page_status = await message.reply(
                f"Обрабатываю страницу {page_index + 1}..."
            )

            try:
                # Acquire rate-limit slot (may wait and notify user)
                if "openrouter" in priority and settings.openrouter_api_key:
                    await _acquire_rate_limit(page_status)

                # Normalize image
                processor = ImageProcessor()
                normalized_bytes, mime_type = processor.normalize_image(image_bytes)

                # Hybrid OCR — only create providers that are in priority
                yandex_provider = get_ocr_provider() if "yandex_ocr" in priority else None
                openrouter_provider = OpenRouterProvider() if "openrouter" in priority and settings.openrouter_api_key else None
                recognizer = HybridRecognizer(
                    yandex_provider=yandex_provider,
                    openrouter_provider=openrouter_provider,
                )
                hybrid_result = await recognizer.recognize(
                    normalized_bytes, mime_type
                )

                passport_data = hybrid_result.passport_data
                modules_used = hybrid_result.modules_used

                # Infer gender from patronymic/surname if not detected
                if not passport_data.gender:
                    inferred = infer_gender(passport_data.middle_name, passport_data.surname)
                    if inferred:
                        passport_data.gender = inferred
                        hybrid_result.field_providers['gender'] = 'inferred'

                # Save to database
                async for session in get_db():
                    repo = PassportRepository(session)

                    quality_score = passport_data.count_filled_fields()

                    record = await repo.create(
                        tg_user_id=tg_user_id,
                        tg_username=tg_username,
                        source_type="pdf_page",
                        source_file_id=source_file_id,
                        source_message_id=source_message_id,
                        source_page_index=page_index,
                        passport_number=passport_data.passport_number,
                        expiry_date=passport_data.expiry_date,
                        surname=passport_data.surname,
                        name=passport_data.name,
                        middle_name=passport_data.middle_name,
                        gender=passport_data.gender,
                        birth_date=passport_data.birth_date,
                        birth_place=passport_data.birth_place,
                        raw_payload=hybrid_result.raw_response,
                        quality_score=quality_score,
                    )

                    # Generate encoded formats
                    format1 = format_passport_type1(passport_data)
                    format2 = format_passport_type2(passport_data)

                    # Build collapsed detail text
                    details = _format_details(
                        passport_data,
                        hybrid_result.field_providers,
                        modules_used,
                        hybrid_result.per_module_data,
                    )

                    response_text = (
                        f"Стр. {page_index + 1}\n"
                        f"<code>{format1}</code>\n"
                        f"<code>{format2}</code>\n"
                        f"<blockquote expandable>"
                        f"{details}"
                        f"</blockquote>"
                    )

                    await page_status.edit_text(response_text, parse_mode="HTML")
                    break

            except Exception as e:
                logger.error("PDF page processing failed", page=page_index, error=str(e))
                await page_status.edit_text(
                    f"Страница {page_index + 1}: ошибка распознавания"
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