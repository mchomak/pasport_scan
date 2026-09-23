"""FastAPI web application for passport OCR."""

import base64
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from config import settings
from services.passport_processing import PassportProcessingService, PassportResult
from services.pdf_processor import PdfProcessor
from services.image_processor import ImageProcessor
from utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Passport OCR", docs_url=None, redoc_url=None)

_processing_service: PassportProcessingService | None = None

_PROVIDER_LABELS = {
    "rupasportread": "Tesseract MRZ",
    "yandex_ocr": "Yandex OCR",
    "openrouter": "OpenRouter LLM",
    "inferred": "Из имени",
    "none": "-",
}

_FIELD_LABELS = {
    "surname": "Фамилия",
    "name": "Имя",
    "middle_name": "Отчество",
    "passport_number": "Серия и номер",
    "birth_date": "Дата рождения",
    "expiry_date": "Срок действия",
    "gender": "Пол",
    "birth_place": "Место рождения",
}


def set_processing_service(service: PassportProcessingService) -> None:
    """Attach the service owned by the application runtime."""
    global _processing_service
    _processing_service = service


def _get_processing_service() -> PassportProcessingService:
    """Return the runtime service, with a standalone-web fallback."""
    global _processing_service
    if _processing_service is None:
        _processing_service = PassportProcessingService()
    return _processing_service


def _build_details(result: PassportResult) -> dict:
    """Keep the existing web detail JSON shape from shared result data."""
    structured = result.structured_details or {}
    fields = structured.get("fields", {})
    providers = structured.get("field_providers", {})

    final_fields = {}
    for field_name, field_label in _FIELD_LABELS.items():
        value = fields.get(field_name)
        provider = providers.get(field_name, "?")
        final_fields[field_name] = {
            "label": field_label,
            "value": str(value) if value is not None and str(value).strip() else None,
            "provider": _PROVIDER_LABELS.get(provider, provider),
        }

    priority = settings.get_module_priority()
    modules_used = structured.get("modules_used", [])
    skipped = [
        _PROVIDER_LABELS.get(module_key, module_key)
        for module_key in priority
        if module_key not in modules_used
    ]

    return {
        "modules": structured.get("modules", []),
        "final": final_fields,
        "skipped": structured.get("skipped", skipped),
    }


async def _process_single_image(image_bytes: bytes) -> dict:
    """Recognize one image through the shared service without persistence."""
    result = await _get_processing_service().recognize_image(image_bytes)
    if not result.success:
        raise RuntimeError(result.error or "recognition failed")

    return {
        "format1": result.format1,
        "format2": result.format2,
        "details": _build_details(result),
        "recognition_state": result.recognition_state,
        "quality_score": result.quality_score,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/recognize")
async def recognize(files: list[UploadFile] = File(...)):
    """Process uploaded files (images or PDFs). Returns results for each page."""
    results = []

    for upload in files:
        content = await upload.read()
        filename = upload.filename or "unknown"
        is_pdf = (
            upload.content_type == "application/pdf"
            or filename.lower().endswith(".pdf")
        )

        try:
            if is_pdf:
                pages = PdfProcessor.extract_pages_as_images(content)
                if not pages:
                    results.append({
                        "filename": filename,
                        "error": "PDF не содержит страниц",
                    })
                    continue
                for page_bytes, page_idx in pages:
                    try:
                        result = await _process_single_image(page_bytes)
                        result["filename"] = f"{filename} (стр. {page_idx + 1})"
                        results.append(result)
                    except Exception as e:
                        logger.error(
                            "Web: PDF page failed",
                            filename=filename,
                            page=page_idx,
                            error=str(e),
                        )
                        results.append({
                            "filename": f"{filename} (стр. {page_idx + 1})",
                            "error": str(e),
                        })
            else:
                result = await _process_single_image(content)
                result["filename"] = filename
                try:
                    norm, _ = ImageProcessor.normalize_image(content)
                    result["thumbnail"] = base64.b64encode(norm).decode("utf-8")
                except Exception:
                    pass
                results.append(result)

        except Exception as e:
            logger.error("Web: file processing failed", filename=filename, error=str(e))
            results.append({"filename": filename, "error": str(e)})

    return {"results": results}
