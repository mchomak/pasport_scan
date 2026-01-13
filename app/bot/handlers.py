"""Telegram bot handlers."""
import io
from typing import Optional
from aiogram import Bot, Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, BufferedInputFile
from aiogram.fsm.context import FSMContext
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.db.database import get_db
from app.db.repository import PassportRepository
from app.ocr.provider import get_ocr_provider
from app.services import ImageProcessor, PdfProcessor, PassportExtractor, ExportService
from app.bot.keyboards import get_export_keyboard
from app.utils.logger import get_logger

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

    status_msg = await message.answer("Обрабатываю фото...")

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

    status_msg = await message.answer("Обрабатываю документ...")

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
    """Process a single image."""
    try:
        # Normalize image
        processor = ImageProcessor()
        normalized_bytes, mime_type = processor.normalize_image(image_bytes)

        # OCR with rotation heuristic
        ocr_provider = get_ocr_provider()
        extractor = PassportExtractor(ocr_provider)
        ocr_result = await extractor.extract_with_rotation(normalized_bytes, mime_type)

        # Save to database
        async for session in get_db():
            repo = PassportRepository(session)

            passport_data = ocr_result.passport_data
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
                raw_payload=ocr_result.raw_response,
                quality_score=quality_score,
            )

            # Format response
            full_name = " ".join(filter(None, [
                passport_data.surname,
                passport_data.name,
                passport_data.middle_name
            ])) or "не найдено"

            response_text = (
                f"✅ Распознавание завершено\n\n"
                f"📝 ID записи: {record.id}\n"
                f"👤 Пользователь: {tg_user_id}\n\n"
                f"📋 Данные паспорта:\n"
                f"ФИО: {full_name}\n"
                f"Серия и номер: {passport_data.passport_number or 'не найдено'}\n"
                f"Дата рождения: {passport_data.birth_date or 'не найдено'}\n"
                f"Место рождения: {passport_data.birth_place or 'не найдено'}\n"
                f"Пол: {passport_data.gender or 'не найдено'}\n"
                f"Выдан: {passport_data.issued_by or 'не найдено'}\n"
                f"Дата выдачи: {passport_data.issue_date or 'не найдено'}\n"
                f"Код подразделения: {passport_data.subdivision_code or 'не найдено'}\n\n"
                f"📊 Заполнено полей: {quality_score}/10"
            )

            await status_msg.edit_text(response_text)
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
            page_status = await message.answer(f"Обрабатываю страницу {page_index + 1}...")

            try:
                # Normalize image
                processor = ImageProcessor()
                normalized_bytes, mime_type = processor.normalize_image(image_bytes)

                # OCR with rotation heuristic
                ocr_provider = get_ocr_provider()
                extractor = PassportExtractor(ocr_provider)
                ocr_result = await extractor.extract_with_rotation(normalized_bytes, mime_type)

                # Save to database
                async for session in get_db():
                    repo = PassportRepository(session)

                    passport_data = ocr_result.passport_data
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
                        raw_payload=ocr_result.raw_response,
                        quality_score=quality_score,
                    )

                    # Format response
                    full_name = " ".join(filter(None, [
                        passport_data.surname,
                        passport_data.name,
                        passport_data.middle_name
                    ])) or "не найдено"

                    response_text = (
                        f"✅ Страница {page_index + 1} обработана\n\n"
                        f"📝 ID записи: {record.id}\n\n"
                        f"📋 Данные:\n"
                        f"ФИО: {full_name}\n"
                        f"Серия и номер: {passport_data.passport_number or 'не найдено'}\n"
                        f"Заполнено полей: {quality_score}/10"
                    )

                    await page_status.edit_text(response_text)
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
