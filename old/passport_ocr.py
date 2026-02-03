#!/usr/bin/env python3
"""
Russian Passport OCR using pytesseract
Extracts all text data from Russian passport images

Структура паспорта РФ:
- Верхняя страница: кем выдан, дата выдачи, код подразделения
- Нижняя страница: фото, ФИО, пол, дата рождения, место рождения, MRZ
"""

import json
import sys
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import pytesseract
from PIL import Image


@dataclass
class PassportData:
    """Structured passport data"""
    # Raw OCR results
    raw_text_rus: str = ""
    raw_text_eng: str = ""
    raw_text_combined: str = ""
    
    # Parsed fields
    surname: Optional[str] = None           # Фамилия (ШУБИН)
    name: Optional[str] = None              # Имя (ДМИТРИЙ)
    patronymic: Optional[str] = None        # Отчество (АНДРЕЕВИЧ)
    sex: Optional[str] = None               # Пол (МУЖ/ЖЕН)
    birth_date: Optional[str] = None        # Дата рождения
    birth_place: Optional[str] = None       # Место рождения
    issue_date: Optional[str] = None        # Дата выдачи
    authority: Optional[str] = None         # Кем выдан
    authority_code: Optional[str] = None    # Код подразделения (500-065)
    series_number: Optional[str] = None     # Серия и номер (46 19 400050)
    
    # MRZ (Machine Readable Zone)
    mrz_line1: Optional[str] = None
    mrz_line2: Optional[str] = None
    mrz_parsed: Optional[Dict[str, Any]] = None
    
    # Zone-specific OCR
    upper_page_text: str = ""
    lower_page_text: str = ""
    data_zone_text: str = ""
    mrz_zone_text: str = ""
    
    # All detected text blocks
    text_blocks: List[Dict] = field(default_factory=list)
    
    # All found symbols (for debugging)
    all_symbols: List[str] = field(default_factory=list)


def preprocess_image(image: np.ndarray, mode: str = 'default') -> np.ndarray:
    """
    Preprocess image for better OCR results
    
    Modes:
    - default: balanced preprocessing
    - high_contrast: for faded text
    - mrz: optimized for machine-readable zone
    - clean: minimal processing for clear images
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if mode == 'clean':
        return gray
    
    if mode == 'mrz':
        # MRZ needs high contrast black on white
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Slight dilation to connect broken characters
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        return binary
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    if mode == 'high_contrast':
        # Additional contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)
    
    # Binarization using adaptive threshold
    binary = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    return binary


def preprocess_for_mrz(image: np.ndarray) -> np.ndarray:
    """Special preprocessing for MRZ zone"""
    return preprocess_image(image, mode='mrz')


def extract_mrz(text: str) -> tuple:
    """Extract MRZ lines from OCR text"""
    # MRZ pattern for Russian passport
    # Line 1: PNRUS + surname + << + name + patronymic (44 chars)
    # Line 2: series+number + check + RUS + birthdate + check + sex + expiry + check + personal + check + final
    
    lines = text.split('\n')
    mrz1, mrz2 = None, None
    
    for line in lines:
        # Clean the line - keep only valid MRZ chars
        cleaned = line.strip().replace(' ', '').upper()
        cleaned = re.sub(r'[^A-Z0-9<]', '', cleaned)
        
        # Skip short lines
        if len(cleaned) < 20:
            continue
        
        # MRZ Line 1: starts with PNRUS or PN<RUS
        if re.match(r'^P[N<]?RUS', cleaned):
            mrz1 = cleaned[:44] if len(cleaned) >= 44 else cleaned
        
        # MRZ Line 2: starts with digits (series+number) and contains RUS
        elif re.match(r'^\d{4,}', cleaned):
            # Clean up common OCR errors
            mrz2_candidate = cleaned[:44] if len(cleaned) >= 44 else cleaned
            if 'RUS' in mrz2_candidate or len(mrz2_candidate) >= 30:
                mrz2 = mrz2_candidate
    
    # Also try to find MRZ-like patterns anywhere in text
    if not mrz2:
        # Look for patterns like 46XXXXXXXXRUS or digits followed by <
        mrz2_pattern = re.search(r'(\d{10,}[A-Z0-9<]{20,})', text.replace(' ', '').replace('\n', ''))
        if mrz2_pattern:
            candidate = re.sub(r'[^A-Z0-9<]', '', mrz2_pattern.group(1).upper())
            if len(candidate) >= 30:
                mrz2 = candidate[:44]
    
    return mrz1, mrz2


def parse_mrz(mrz1: str, mrz2: str) -> dict:
    """Parse MRZ data into structured fields"""
    result = {}
    
    if mrz1:
        # Remove P, N, RUS prefix
        name_part = re.sub(r'^P[N<]*RUS', '', mrz1)
        # Split by << (double chevron separates surname from given names)
        parts = name_part.split('<<')
        if len(parts) >= 1:
            result['mrz_surname'] = parts[0].replace('<', ' ').strip()
        if len(parts) >= 2:
            names = parts[1].replace('<', ' ').strip().split()
            if names:
                result['mrz_name'] = names[0]
            if len(names) > 1:
                result['mrz_patronymic'] = names[1]
    
    if mrz2:
        # Parse Line 2: SSSSNNNNNNCRRRYYMMDDCGYYMMDDCPPPPPPPPPPPPPPPC
        # SSSS = series (4 digits)
        # NNNNNN = number (6 digits)
        # C = check digit
        # RRR = nationality (RUS)
        # YYMMDD = birth date
        # C = check digit
        # G = gender (M/F)
        # YYMMDD = expiry date
        # C = check digit
        # etc.
        
        if len(mrz2) >= 10:
            series = mrz2[:4]
            number = mrz2[4:10]
            result['mrz_series_number'] = f"{series[:2]} {series[2:]} {number}"
        
        if len(mrz2) >= 20:
            birth = mrz2[13:19]  # YYMMDD
            if birth.isdigit():
                yy, mm, dd = birth[:2], birth[2:4], birth[4:6]
                # Determine century (00-30 = 2000s, 31-99 = 1900s for passports)
                year = int(yy)
                century = '20' if year <= 30 else '19'
                result['mrz_birth_date'] = f"{dd}.{mm}.{century}{yy}"
        
        if len(mrz2) >= 21:
            sex = mrz2[20]
            result['mrz_sex'] = 'МУЖ' if sex == 'M' else 'ЖЕН' if sex == 'F' else sex
    
    return result


def parse_russian_text(text: str) -> dict:
    """Parse Russian text to extract passport fields"""
    result = {}
    
    # Normalize text
    text_normalized = text.upper()
    
    # Date patterns (DD.MM.YYYY)
    date_pattern = r'\d{2}[.\-/]\d{2}[.\-/]\d{4}'
    dates = re.findall(date_pattern, text)
    
    # Authority code pattern (XXX-XXX)
    code_pattern = r'(\d{3})[-‐–—](\d{3})'
    code_match = re.search(code_pattern, text)
    if code_match:
        result['authority_code'] = f"{code_match.group(1)}-{code_match.group(2)}"
    
    # Series and number patterns
    # Pattern 1: XX XX XXXXXX (with spaces)
    series_match = re.search(r'\b(\d{2})\s+(\d{2})\s+(\d{6})\b', text)
    if series_match:
        result['series_number'] = f"{series_match.group(1)} {series_match.group(2)} {series_match.group(3)}"
    else:
        # Pattern 2: XXXX XXXXXX (combined series)
        series_match = re.search(r'\b(\d{4})\s+(\d{6})\b', text)
        if series_match:
            s = series_match.group(1)
            result['series_number'] = f"{s[:2]} {s[2:]} {series_match.group(2)}"
    
    # Sex - look for МУЖ or ЖЕН patterns
    if re.search(r'\bМУЖ[.\s]', text_normalized) or 'МУЖ' in text_normalized:
        result['sex'] = 'МУЖ'
    elif re.search(r'\bЖЕН[.\s]', text_normalized) or 'ЖЕН' in text_normalized:
        result['sex'] = 'ЖЕН'
    
    # Known Russian names patterns - common surnames and first names
    # Look for capitalized Cyrillic words that look like names
    
    # Common name patterns found near labels or standalone
    name_patterns = [
        # Full name pattern: SURNAME NAME PATRONYMIC (all caps)
        r'([А-ЯЁ]{3,})\s+([А-ЯЁ]{3,})\s+([А-ЯЁ]{5,}ИЧ|[А-ЯЁ]{5,}НА)',
        # Patronymic pattern (ending in -ИЧ or -НА/-ВНА)
        r'\b([А-ЯЁ]{5,}(?:ЕВИЧ|ОВИЧ|ИЧ|ЕВНА|ОВНА|ИЧНА|НА))\b',
    ]
    
    # Find patronymic first (most distinctive)
    patronymic_match = re.search(r'\b([А-ЯЁ]{5,}(?:ЕЕВИЧ|ЕВИЧ|ОВИЧ|ИЧ))\b', text_normalized)
    if patronymic_match:
        result['patronymic'] = patronymic_match.group(1)
    else:
        # Female patronymic
        patronymic_match = re.search(r'\b([А-ЯЁ]{5,}(?:ЕВНА|ОВНА))\b', text_normalized)
        if patronymic_match:
            result['patronymic'] = patronymic_match.group(1)
    
    # Common first names dictionary (for matching)
    common_male_names = ['ДМИТРИЙ', 'АЛЕКСАНДР', 'СЕРГЕЙ', 'АНДРЕЙ', 'АЛЕКСЕЙ', 'ИВАН', 
                         'МИХАИЛ', 'НИКОЛАЙ', 'ВЛАДИМИР', 'ПАВЕЛ', 'АРТЁМ', 'МАКСИМ',
                         'ДЕНИС', 'ЕГОР', 'КИРИЛЛ', 'ИЛЬЯ', 'РОМАН', 'ВИКТОР', 'ОЛЕГ']
    common_female_names = ['МАРИЯ', 'АННА', 'ЕЛЕНА', 'ОЛЬГА', 'НАТАЛЬЯ', 'ТАТЬЯНА',
                           'ИРИНА', 'СВЕТЛАНА', 'ЕКАТЕРИНА', 'ЮЛИЯ', 'АНАСТАСИЯ']
    
    all_names = common_male_names + common_female_names
    
    # Search for common first names in text
    for name in all_names:
        if name in text_normalized:
            result['name'] = name
            break
    
    # Find surname - typically precedes the name or is near it
    # Look for capitalized words that are NOT known labels
    labels = {'РОССИЙСКАЯ', 'ФЕДЕРАЦИЯ', 'ПАСПОРТ', 'ВЫДАН', 'ФАМИЛИЯ', 'ИМЯ', 
              'ОТЧЕСТВО', 'ПОЛ', 'ДАТА', 'РОЖДЕНИЯ', 'МЕСТО', 'ОБЛАСТЬ', 
              'РОССИЯ', 'МОСКОВСКАЯ', 'МОСКОВСКОЙ', 'ОБЛАСТИ', 'МВД', 'УМВД',
              'ГУ', 'УФМС', 'ФМС', 'ПОДРАЗДЕЛЕНИЯ', 'КОД', 'ГРАЖДАНИНА',
              'ЛИЧНАЯ', 'ПОДПИСЬ', 'ЛИЧНЫЙ', 'КОД', 'ВЫДАЧИ'}
    
    # Extract all potential surname candidates (Cyrillic words 4+ chars, not in labels/names)
    cyrillic_words = re.findall(r'\b([А-ЯЁ]{4,})\b', text_normalized)
    name_found = result.get('name', '')
    patronymic_found = result.get('patronymic', '')
    
    for word in cyrillic_words:
        if (word not in labels and 
            word != name_found and 
            word != patronymic_found and
            word not in all_names and
            len(word) >= 4 and
            not word.endswith('ИЧ') and
            not word.endswith('ВНА') and
            not word.endswith('ЕВН') and
            'МОСКОВ' not in word):
            # Could be surname
            if not result.get('surname'):
                result['surname'] = word
                break
    
    # Lines-based parsing for authority and birth place
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        line_upper = line_clean.upper()
        
        # Authority (ГУ МВД, УФМС, etc.)
        if 'МВД' in line_upper or 'УФМС' in line_upper or 'ФМС' in line_upper:
            if len(line_clean) > 10:
                result['authority'] = line_clean
        
        # Birth place (ГОР. / Г. / ГОРОД / city names)
        birth_place_match = re.search(r'(?:ГОР\.?|Г\.)\s*([А-ЯЁа-яё]+)', line)
        if birth_place_match:
            place = birth_place_match.group(1).strip().upper()
            if place and place not in labels:
                result['birth_place'] = place
        
        # Check for МОСКВА specifically
        if 'МОСКВА' in line_upper and 'МОСКОВСК' not in line_upper:
            if not result.get('birth_place'):
                result['birth_place'] = 'МОСКВА'
    
    # Assign dates heuristically
    if dates:
        for d in dates:
            parts = re.split(r'[.\-/]', d)
            if len(parts) == 3:
                try:
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    # Birth date: typically 1950-2020
                    if 1950 <= year <= 2020:
                        if not result.get('birth_date'):
                            result['birth_date'] = d
                    # Issue date: typically 2010-2030
                    elif 2010 <= year <= 2030:
                        if not result.get('issue_date'):
                            result['issue_date'] = d
                except ValueError:
                    continue
    
    return result


def extract_text_blocks(image: np.ndarray, lang: str = 'rus+eng') -> list:
    """Extract individual text blocks with their positions"""
    data = pytesseract.image_to_data(
        image, 
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config='--psm 6'
    )
    
    blocks = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        if text and conf > 20:  # Filter very low confidence
            blocks.append({
                'text': text,
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'confidence': conf,
                'block_num': data['block_num'][i],
                'line_num': data['line_num'][i]
            })
    
    return blocks


def ocr_zone(image: np.ndarray, lang: str, psm: int = 6, whitelist: str = None) -> str:
    """OCR a specific zone with custom settings"""
    config = f'--oem 3 --psm {psm} -l {lang}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'
    
    try:
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()
    except Exception as e:
        return f"[OCR Error: {e}]"


def ocr_passport(image_path: str) -> PassportData:
    """Main function to OCR a Russian passport image"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Initialize result
    passport = PassportData()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define zones based on typical passport layout
    # The uploaded image shows two pages: upper (issue info) and lower (personal data + MRZ)
    
    # Split into upper and lower halves (for 2-page layout)
    mid_point = height // 2
    upper_page = image[:mid_point, :]
    lower_page = image[mid_point:, :]
    
    # Process full image first
    processed_full = preprocess_image(image)
    
    # === OCR with Russian language ===
    passport.raw_text_rus = ocr_zone(processed_full, 'rus', psm=6)
    
    # === OCR with English (for MRZ and Latin chars) ===
    passport.raw_text_eng = ocr_zone(processed_full, 'eng', psm=6)
    
    # === Combined OCR ===
    passport.raw_text_combined = ocr_zone(processed_full, 'rus+eng', psm=6)
    
    # === Zone-specific OCR ===
    
    # Upper page (issue authority, date, code)
    processed_upper = preprocess_image(upper_page)
    passport.upper_page_text = ocr_zone(processed_upper, 'rus', psm=6)
    
    # Lower page (personal data)
    processed_lower = preprocess_image(lower_page)
    passport.lower_page_text = ocr_zone(processed_lower, 'rus+eng', psm=6)
    
    # Data zone (right side of lower page where text data is)
    lower_height, lower_width = lower_page.shape[:2]
    # Personal data is typically in the right 70% of the lower page
    data_zone = lower_page[:int(lower_height * 0.7), int(lower_width * 0.25):]
    processed_data = preprocess_image(data_zone)
    passport.data_zone_text = ocr_zone(processed_data, 'rus', psm=6)
    
    # MRZ zone (bottom ~20% of lower page)
    mrz_zone = lower_page[int(lower_height * 0.75):, :]
    processed_mrz = preprocess_image(mrz_zone, mode='mrz')
    
    # OCR MRZ with specific whitelist
    mrz_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    passport.mrz_zone_text = ocr_zone(processed_mrz, 'eng', psm=6, whitelist=mrz_whitelist)
    
    # Also try without whitelist for comparison
    mrz_text_alt = ocr_zone(processed_mrz, 'eng', psm=6)
    
    # Extract text blocks for detailed analysis
    passport.text_blocks = extract_text_blocks(processed_full)
    
    # Collect all symbols found
    all_text = passport.raw_text_rus + passport.raw_text_eng + passport.mrz_zone_text
    passport.all_symbols = list(set(c for c in all_text if c.strip()))
    
    # === Parse MRZ ===
    # Combine all text sources that might contain MRZ
    combined_mrz_text = '\n'.join([
        passport.mrz_zone_text or '',
        mrz_text_alt or '',
        passport.raw_text_eng or '',
        passport.lower_page_text or '',
        passport.raw_text_combined or ''
    ])
    mrz1, mrz2 = extract_mrz(combined_mrz_text)
    passport.mrz_line1 = mrz1
    passport.mrz_line2 = mrz2
    
    if mrz1 or mrz2:
        passport.mrz_parsed = parse_mrz(mrz1, mrz2)
    
    # === Parse Russian text ===
    combined_text = passport.raw_text_rus + '\n' + passport.data_zone_text + '\n' + passport.lower_page_text
    parsed = parse_russian_text(combined_text)
    
    # Assign parsed fields
    passport.surname = parsed.get('surname')
    passport.name = parsed.get('name')
    passport.patronymic = parsed.get('patronymic')
    passport.sex = parsed.get('sex')
    passport.birth_date = parsed.get('birth_date')
    passport.birth_place = parsed.get('birth_place')
    passport.issue_date = parsed.get('issue_date')
    passport.authority = parsed.get('authority')
    passport.authority_code = parsed.get('authority_code')
    passport.series_number = parsed.get('series_number')
    
    # Fill in from MRZ if visual OCR missed something
    if passport.mrz_parsed:
        if not passport.surname and passport.mrz_parsed.get('mrz_surname'):
            passport.surname = passport.mrz_parsed['mrz_surname']
        if not passport.name and passport.mrz_parsed.get('mrz_name'):
            passport.name = passport.mrz_parsed['mrz_name']
        if not passport.patronymic and passport.mrz_parsed.get('mrz_patronymic'):
            passport.patronymic = passport.mrz_parsed['mrz_patronymic']
        if not passport.birth_date and passport.mrz_parsed.get('mrz_birth_date'):
            passport.birth_date = passport.mrz_parsed['mrz_birth_date']
        if not passport.sex and passport.mrz_parsed.get('mrz_sex'):
            passport.sex = passport.mrz_parsed['mrz_sex']
        if not passport.series_number and passport.mrz_parsed.get('mrz_series_number'):
            passport.series_number = passport.mrz_parsed['mrz_series_number']
    
    return passport


def save_result(passport: PassportData, output_path: str):
    """Save passport data to JSON file"""
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    data = asdict(passport)
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    image_path = "C:\\Users\\McHomak\\Projects\\pasport_scan\\data\\2.jpg"
    output_path = "./data/result.json"
    
    print(f"Processing: {image_path}")
    
    try:
        passport_data = ocr_passport(image_path)
        save_result(passport_data, output_path)
        
        # Print summary
        print("\n=== Extracted Data ===")
        print(f"Surname: {passport_data.surname}")
        print(f"Name: {passport_data.name}")
        print(f"Patronymic: {passport_data.patronymic}")
        print(f"Sex: {passport_data.sex}")
        print(f"Birth Date: {passport_data.birth_date}")
        print(f"Birth Place: {passport_data.birth_place}")
        print(f"Issue Date: {passport_data.issue_date}")
        print(f"Authority: {passport_data.authority}")
        print(f"Authority Code: {passport_data.authority_code}")
        print(f"Series/Number: {passport_data.series_number}")
        print(f"MRZ Line 1: {passport_data.mrz_line1}")
        print(f"MRZ Line 2: {passport_data.mrz_line2}")
        print(f"\nTotal text blocks found: {len(passport_data.text_blocks)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
