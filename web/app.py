"""FastAPI web application for passport OCR."""
import base64
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config import settings
from ocr.provider import get_ocr_provider
from ocr.openrouter import OpenRouterProvider
from ocr.hybrid import HybridRecognizer
from services.image_processor import ImageProcessor
from services.pdf_processor import PdfProcessor
from utils.logger import get_logger
from utils.passport_formatter import (
    format_passport_type1,
    format_passport_type2,
    infer_gender,
)
from utils.rate_limiter import MinuteRateLimiter

logger = get_logger(__name__)

app = FastAPI(title="Passport OCR", docs_url=None, redoc_url=None)

# Share the same RPM limiter across web requests
_openrouter_limiter = MinuteRateLimiter(rpm=settings.openrouter_rpm)

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


def _build_details(passport_data, field_providers, per_module_data) -> list[dict]:
    """Build detail blocks for each module."""
    priority = settings.get_module_priority()
    modules = []
    for module_key in priority:
        if module_key in per_module_data:
            data = per_module_data[module_key]
            fields = {}
            for field_name, field_label in _FIELD_LABELS.items():
                val = getattr(data, field_name, None)
                fields[field_name] = {
                    "label": field_label,
                    "value": str(val) if val is not None and str(val).strip() else None,
                }
            modules.append({
                "key": module_key,
                "label": _PROVIDER_LABELS.get(module_key, module_key),
                "fields": fields,
            })

    # Final merged
    final_fields = {}
    for field_name, field_label in _FIELD_LABELS.items():
        val = getattr(passport_data, field_name, None)
        provider = field_providers.get(field_name, "?")
        final_fields[field_name] = {
            "label": field_label,
            "value": str(val) if val is not None and str(val).strip() else None,
            "provider": _PROVIDER_LABELS.get(provider, provider),
        }

    skipped = [
        _PROVIDER_LABELS.get(m, m)
        for m in priority
        if m not in per_module_data
    ]

    return {
        "modules": modules,
        "final": final_fields,
        "skipped": skipped,
    }


async def _process_single_image(image_bytes: bytes) -> dict:
    """Process one image through the hybrid OCR pipeline. Returns JSON-ready dict."""
    priority = settings.get_module_priority()

    # Rate limit
    if "openrouter" in priority and settings.openrouter_api_key:
        if _openrouter_limiter.is_enabled:
            waited = await _openrouter_limiter.acquire()
            if waited > 0:
                logger.info("Web: rate limiter waited %.1f s", waited)

    # Normalize
    processor = ImageProcessor()
    normalized_bytes, mime_type = processor.normalize_image(image_bytes)

    # Recognize
    yandex_provider = get_ocr_provider() if "yandex_ocr" in priority else None
    openrouter_provider = (
        OpenRouterProvider()
        if "openrouter" in priority and settings.openrouter_api_key
        else None
    )
    recognizer = HybridRecognizer(
        yandex_provider=yandex_provider,
        openrouter_provider=openrouter_provider,
    )
    hybrid_result = await recognizer.recognize(normalized_bytes, mime_type)
    passport_data = hybrid_result.passport_data

    # Infer gender
    if not passport_data.gender:
        inferred = infer_gender(passport_data.middle_name, passport_data.surname)
        if inferred:
            passport_data.gender = inferred
            hybrid_result.field_providers['gender'] = 'inferred'

    format1 = format_passport_type1(passport_data)
    format2 = format_passport_type2(passport_data)

    details = _build_details(
        passport_data,
        hybrid_result.field_providers,
        hybrid_result.per_module_data,
    )

    return {
        "format1": format1,
        "format2": format2,
        "details": details,
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
                pdf_processor = PdfProcessor()
                pages = pdf_processor.extract_pages_as_images(content)
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
                        logger.error("Web: PDF page failed", filename=filename, page=page_idx, error=str(e))
                        results.append({
                            "filename": f"{filename} (стр. {page_idx + 1})",
                            "error": str(e),
                        })
            else:
                result = await _process_single_image(content)
                result["filename"] = filename
                # Return base64 thumbnail for detail view
                try:
                    processor = ImageProcessor()
                    norm, _ = processor.normalize_image(content)
                    result["thumbnail"] = base64.b64encode(norm).decode("utf-8")
                except Exception:
                    pass
                results.append(result)

        except Exception as e:
            logger.error("Web: file processing failed", filename=filename, error=str(e))
            results.append({"filename": filename, "error": str(e)})

    return {"results": results}
