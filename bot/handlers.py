"""Telegram bot handlers."""
import io
from typing import Optional
from aiogram import Bot, Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, BufferedInputFile
from aiogram.fsm.context import FSMContext
from sqlalchemy.ext.asyncio import AsyncSession
from config import settings
from db.database import get_db
from db.repository import PassportRepository
from ocr.provider import get_ocr_provider
from ocr.hybrid import HybridRecognizer
from services import ImageProcessor, PdfProcessor, ExportService
from bot.keyboards import get_export_keyboard
from utils.logger import get_logger
from utils.passport_formatter import (
    format_passport_type1,
    format_passport_type2,
    calculate_expiry_date,
    infer_gender,
)

logger = get_logger(__name__)
router = Router()


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
        await callback.message.edit_text(
            f"Ошибка при формировании выгрузки: {str(e)}"
        )

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

    status_msg = await message.reply("Обрабатываю фото (гибридное распознавание)...")

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
        await status_msg.edit_text(
            "Произошла ошибка при обработке фото. "
            "Пожалуйста, попробуйте другое изображение."
        )


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

    status_msg = await message.reply("Обрабатываю документ (гибридное распознавание)...")

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
        await status_msg.edit_text(
            "Произошла ошибка при обработке документа. "
            "Пожалуйста, попробуйте другой файл."
        )


# --- Processing Functions ---

_PROVIDER_LABELS = {
    'rupasportread': 'Tesseract MRZ',
    'easyocr': 'EasyOCR',
    'yandex_ocr': 'Yandex OCR',
    'inferred': 'Из имени',
    'none': '-',
}

_PROVIDER_ORDER = ['rupasportread', 'easyocr', 'yandex_ocr']

_FIELD_LABELS = {
    'surname': 'Фамилия',
    'name': 'Имя',
    'middle_name': 'Отчество',
    'passport_number': 'Серия и номер',
    'birth_date': 'Дата рождения',
    'issue_date': 'Дата выдачи',
    'gender': 'Пол',
    'birth_place': 'Место рождения',
    'issued_by': 'Кем выдан',
    'subdivision_code': 'Код подразделения',
}


def _format_modules_used(modules: list[str]) -> str:
    """Format module names for display."""
    return ' + '.join(_PROVIDER_LABELS.get(m, m) for m in modules)


def _format_module_block(module_key: str, data) -> str:
    """Format a single module's full results (all fields)."""
    label = _PROVIDER_LABELS.get(module_key, module_key)
    lines = [f"[{label}]"]
    found_any = False
    for field_name, field_label in _FIELD_LABELS.items():
        val = getattr(data, field_name, None)
        if val is not None and str(val).strip():
            lines.append(f"  + {field_label}: {val}")
            found_any = True
        else:
            lines.append(f"  - {field_label}: ---")
    if not found_any:
        lines.append("  (ничего не найдено)")
    return '\n'.join(lines)


def _format_provider_details(
    passport_data,
    field_providers: dict,
    modules_used: list[str],
    per_module_data: dict,
) -> str:
    """Build full per-module data blocks + final merged result."""
    lines: list[str] = []

    # Per-module blocks — show what each module found independently
    for module_key in _PROVIDER_ORDER:
        if module_key in per_module_data:
            lines.append(_format_module_block(module_key, per_module_data[module_key]))
            lines.append("")

    # Modules that were skipped
    skipped = [
        _PROVIDER_LABELS.get(m, m) for m in _PROVIDER_ORDER
        if m not in per_module_data
    ]
    if skipped:
        lines.append(f"Пропущены: {', '.join(skipped)}")
        lines.append("")

    # Final merged result with source attribution
    lines.append("[Итог (объединённые данные)]")
    for field_name, field_label in _FIELD_LABELS.items():
        val = getattr(passport_data, field_name, None)
        if val is not None and str(val).strip():
            src = _PROVIDER_LABELS.get(field_providers.get(field_name, '?'), '?')
            lines.append(f"  {field_label}: {val}  ({src})")
        else:
            lines.append(f"  {field_label}: --- не найдено")

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
        # Normalize image
        processor = ImageProcessor()
        normalized_bytes, mime_type = processor.normalize_image(image_bytes)

        # Hybrid OCR recognition
        yandex_provider = get_ocr_provider()
        recognizer = HybridRecognizer(yandex_provider=yandex_provider)
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
                issued_by=passport_data.issued_by,
                issue_date=passport_data.issue_date,
                subdivision_code=passport_data.subdivision_code,
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

            # Check expiry date correction
            _, expiry_corrected, years_added = calculate_expiry_date(
                passport_data.birth_date, passport_data.issue_date,
                passport_data.passport_number)
            expiry_warning = ""
            if expiry_corrected:
                expiry_warning = (
                    "\n--- ВНИМАНИЕ ---\n"
                    "Дата действия паспорта была в прошлом. "
                    f"Дата скорректирована (+{years_added} лет) в форматах ниже.\n"
                )

            # Build detailed provider attribution
            provider_details = _format_provider_details(
                passport_data,
                hybrid_result.field_providers,
                modules_used,
                hybrid_result.per_module_data,
            )

            response_text = (
                f"Распознавание завершено\n\n"
                f"ID записи: {record.id}\n"
                f"Заполнено полей: {quality_score}/10\n\n"
                f"--- Детализация по провайдерам ---\n"
                f"{provider_details}\n"
                f"{expiry_warning}\n"
                f"--- Закодированные форматы ---\n"
                f"<code>{format1}</code>\n\n"
                f"<code>{format2}</code>"
            )

            await status_msg.edit_text(response_text, parse_mode="HTML")
            break

    except Exception as e:
        logger.error("Image processing failed", error=str(e))
        await status_msg.edit_text(
            "Не удалось распознать паспорт. "
            "Пожалуйста, попробуйте более чёткий снимок."
        )


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
                f"Обрабатываю страницу {page_index + 1} (гибридное распознавание)..."
            )

            try:
                # Normalize image
                processor = ImageProcessor()
                normalized_bytes, mime_type = processor.normalize_image(image_bytes)

                # Hybrid OCR
                yandex_provider = get_ocr_provider()
                recognizer = HybridRecognizer(yandex_provider=yandex_provider)
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
                        issued_by=passport_data.issued_by,
                        issue_date=passport_data.issue_date,
                        subdivision_code=passport_data.subdivision_code,
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

                    # Check expiry date correction
                    _, expiry_corrected, years_added = calculate_expiry_date(
                        passport_data.birth_date, passport_data.issue_date,
                        passport_data.passport_number)
                    expiry_warning = ""
                    if expiry_corrected:
                        expiry_warning = (
                            "\n--- ВНИМАНИЕ ---\n"
                            "Дата действия паспорта была в прошлом. "
                            f"Дата скорректирована (+{years_added} лет) в форматах ниже.\n"
                        )

                    # Build detailed provider attribution
                    provider_details = _format_provider_details(
                        passport_data,
                        hybrid_result.field_providers,
                        modules_used,
                        hybrid_result.per_module_data,
                    )

                    response_text = (
                        f"Страница {page_index + 1} обработана\n\n"
                        f"ID записи: {record.id}\n"
                        f"Заполнено полей: {quality_score}/10\n\n"
                        f"--- Детализация по провайдерам ---\n"
                        f"{provider_details}\n"
                        f"{expiry_warning}\n"
                        f"--- Закодированные форматы ---\n"
                        f"<code>{format1}</code>\n\n"
                        f"<code>{format2}</code>"
                    )

                    await page_status.edit_text(response_text, parse_mode="HTML")
                    break

            except Exception as e:
                logger.error("PDF page processing failed", page=page_index, error=str(e))
                await page_status.edit_text(
                    f"❌ Страница {page_index + 1}: ошибка распознавания"
                )

        await status_msg.edit_text(f"✅ PDF обработан ({len(pages)} стр.)")

    except Exception as e:
        logger.error("PDF processing failed", error=str(e))
        await status_msg.edit_text(
            "Не удалось обработать PDF. "
            "Пожалуйста, попробуйте другой файл."
        )