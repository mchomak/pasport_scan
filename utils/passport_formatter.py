"""Passport data formatting utilities."""
from datetime import date
from typing import Optional
from ocr.models import PassportData


def transliterate_to_latin(text: Optional[str]) -> str:
    """
    Транслитерация кириллицы в латиницу.

    Args:
        text: Текст на кириллице

    Returns:
        Текст латиницей
    """
    if not text:
        return "unknown"

    # Таблица транслитерации (ГОСТ 7.79-2000, система Б)
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
        'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
        'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
        'Ф': 'F', 'Х': 'H', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
        'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
    }

    result = []
    for char in text:
        if char in translit_map:
            result.append(translit_map[char])
        else:
            result.append(char)

    return ''.join(result)


def get_country_code(birth_place: Optional[str]) -> str:
    """
    Определяет код страны по месту рождения.

    Args:
        birth_place: Место рождения

    Returns:
        Код страны (uz, td, kg, ru)
    """
    if not birth_place:
        return "ru"

    birth_place_lower = birth_place.lower()

    # Узбекистан
    if any(word in birth_place_lower for word in ["узбек", "uzbek", "ташкент", "самарканд", "бухара"]):
        return "uz"

    # Таджикистан
    if any(word in birth_place_lower for word in ["таджик", "tajik", "душанбе", "худжанд"]):
        return "td"

    # Киргизия
    if any(word in birth_place_lower for word in ["кирги", "кыргы", "kyrgyz", "бишкек", "ош"]):
        return "kg"

    # По умолчанию Россия
    return "ru"


def get_document_type(passport_number: Optional[str]) -> str:
    """
    Определяет тип документа.

    Args:
        passport_number: Номер паспорта

    Returns:
        Тип документа (NP, PSP, PS)
    """
    if not passport_number:
        return "PS"

    # Российский внутренний паспорт: серия 4 цифры + номер 6 цифр (4619709685)
    if len(passport_number.replace(" ", "")) == 10:
        return "PS"

    # Заграничный паспорт РФ: начинается с цифр (72, 73, 74...)
    if passport_number and passport_number[0].isdigit() and len(passport_number) >= 9:
        return "PSP"

    # Национальный паспорт других стран
    return "NP"


def get_gender_code(gender: Optional[str]) -> str:
    """
    Преобразует пол в код.

    Args:
        gender: Пол (муж/жен)

    Returns:
        Код пола (m/f)
    """
    if not gender:
        return "m"

    gender_lower = gender.lower()
    if "жен" in gender_lower or "female" in gender_lower or "f" == gender_lower:
        return "f"

    return "m"


def format_date_short(d: Optional[date]) -> str:
    """
    Форматирует дату в короткий формат DDMMYY.

    Args:
        d: Дата

    Returns:
        Строка вида "100805"
    """
    if not d:
        return "000000"

    return d.strftime("%d%m%y")


def format_date_long(d: Optional[date]) -> str:
    """
    Форматирует дату в длинный формат DDmmmYY.

    Args:
        d: Дата

    Returns:
        Строка вида "10aug05"
    """
    if not d:
        return "00xxx00"

    months = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
        7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
    }

    day = d.strftime("%d")
    month = months.get(d.month, "xxx")
    year = d.strftime("%y")

    return f"{day}{month}{year}"


def calculate_expiry_date(birth_date: Optional[date], issue_date: Optional[date]) -> Optional[date]:
    """
    Рассчитывает срок действия российского паспорта.

    Паспорт РФ действует:
    - До 20 лет (выдается в 14)
    - До 45 лет (выдается в 20)
    - Бессрочно после 45 лет

    Args:
        birth_date: Дата рождения
        issue_date: Дата выдачи

    Returns:
        Дата окончания срока действия или None
    """
    if not birth_date or not issue_date:
        return None

    age_at_issue = issue_date.year - birth_date.year

    # Первый паспорт (14-20 лет) - действует до 20 лет
    if age_at_issue < 20:
        expiry = date(birth_date.year + 20, birth_date.month, birth_date.day)
        return expiry

    # Второй паспорт (20-45 лет) - действует до 45 лет
    if age_at_issue < 45:
        expiry = date(birth_date.year + 45, birth_date.month, birth_date.day)
        return expiry

    # Третий паспорт (после 45) - бессрочный, ставим +20 лет для формата
    expiry = date(issue_date.year + 20, issue_date.month, issue_date.day)
    return expiry


def format_passport_type1(data: PassportData) -> str:
    """
    Формат 1: country/number/country/birthdate/gender/expiry/surname/name
    Пример: uz/fa2971721/uz/10aug05/m/06jun26/yafarov/amir

    Args:
        data: Данные паспорта

    Returns:
        Отформатированная строка
    """
    country = get_country_code(data.birth_place)
    number = (data.passport_number or "0000000000").replace(" ", "").lower()
    birth_date = format_date_long(data.birth_date)
    gender = get_gender_code(data.gender)
    expiry_date = calculate_expiry_date(data.birth_date, data.issue_date)
    expiry = format_date_long(expiry_date)
    surname = transliterate_to_latin(data.surname).lower()
    name = transliterate_to_latin(data.name).lower()

    return f"{country}/{number}/{country}/{birth_date}/{gender}/{expiry}/{surname}/{name}"


def format_passport_type2(data: PassportData) -> str:
    """
    Формат 2: -surname name birthdate+gender/country/doc_type number/expiry
    Пример: -yafarov amir 100805+m/uz/NP fa2971721/060626

    Args:
        data: Данные паспорта

    Returns:
        Отформатированная строка
    """
    surname = transliterate_to_latin(data.surname).lower()
    name = transliterate_to_latin(data.name).lower()
    birth_date = format_date_short(data.birth_date)
    gender = get_gender_code(data.gender)
    country = get_country_code(data.birth_place)
    doc_type = get_document_type(data.passport_number)
    number = (data.passport_number or "0000000000").replace(" ", "").lower()
    expiry_date = calculate_expiry_date(data.birth_date, data.issue_date)
    expiry = format_date_short(expiry_date)

    return f"-{surname} {name} {birth_date}+{gender}/{country}/{doc_type} {number}/{expiry}"