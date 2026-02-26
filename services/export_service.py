"""Data export service for admin users."""
import io
import csv
from typing import List, Literal
import pandas as pd
from db.models import PassportRecord
from utils.logger import get_logger

logger = get_logger(__name__)


class ExportService:
    """Service for exporting passport records."""

    @staticmethod
    def export_csv(records: List[PassportRecord]) -> bytes:
        """
        Export records to CSV format.

        Args:
            records: List of passport records

        Returns:
            CSV file bytes
        """
        logger.info("Exporting to CSV", record_count=len(records))

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        headers = [
            "ID",
            "Created At",
            "Telegram User ID",
            "Telegram Username",
            "Source Type",
            "Page Index",
            "Passport Number",
            "Expiry Date",
            "Surname",
            "Name",
            "Middle Name",
            "Gender",
            "Birth Date",
            "Birth Place",
            "Quality Score",
        ]
        writer.writerow(headers)

        # Write data
        for record in records:
            row = [
                str(record.id),
                record.created_at.isoformat() if record.created_at else "",
                record.tg_user_id,
                record.tg_username or "",
                record.source_type,
                record.source_page_index if record.source_page_index is not None else "",
                record.passport_number or "",
                record.expiry_date.isoformat() if record.expiry_date else "",
                record.surname or "",
                record.name or "",
                record.middle_name or "",
                record.gender or "",
                record.birth_date.isoformat() if record.birth_date else "",
                record.birth_place or "",
                record.quality_score,
            ]
            writer.writerow(row)

        # Get bytes
        csv_bytes = output.getvalue().encode("utf-8-sig")  # UTF-8 with BOM for Excel
        logger.info("CSV export completed", size_bytes=len(csv_bytes))

        return csv_bytes

    @staticmethod
    def export_excel(records: List[PassportRecord]) -> bytes:
        """
        Export records to Excel format.

        Args:
            records: List of passport records

        Returns:
            Excel file bytes
        """
        logger.info("Exporting to Excel", record_count=len(records))

        # Prepare data for DataFrame
        data = []
        for record in records:
            data.append({
                "ID": str(record.id),
                "Created At": record.created_at.isoformat() if record.created_at else "",
                "Telegram User ID": record.tg_user_id,
                "Telegram Username": record.tg_username or "",
                "Source Type": record.source_type,
                "Page Index": record.source_page_index if record.source_page_index is not None else "",
                "Passport Number": record.passport_number or "",
                "Expiry Date": record.expiry_date.isoformat() if record.expiry_date else "",
                "Surname": record.surname or "",
                "Name": record.name or "",
                "Middle Name": record.middle_name or "",
                "Gender": record.gender or "",
                "Birth Date": record.birth_date.isoformat() if record.birth_date else "",
                "Birth Place": record.birth_place or "",
                "Quality Score": record.quality_score,
            })

        # Create DataFrame
        df = pd.DataFrame(data)

        # Write to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='passports', index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets['passports']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter

                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass

                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        excel_bytes = output.getvalue()
        logger.info("Excel export completed", size_bytes=len(excel_bytes))

        return excel_bytes
