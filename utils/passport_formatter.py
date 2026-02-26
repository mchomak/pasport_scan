"""Passport data formatting utilities."""
from datetime import date
from typing import Optional
from ocr.models import PassportData
from config import settings


def transliterate_to_latin(text: Optional[str]) -> str:
    """Транслитерация кириллицы в латиницу."""
    if not text:
        return "unknown"

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


def get_country_code(
    birth_place: Optional[str] = None,
    passport_number: Optional[str] = None,
    surname: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Определяет код страны по месту рождения, серии паспорта и ФИО."""
    if birth_place:
        bp = birth_place.lower()
        if any(w in bp for w in [
            "узбек", "uzbek", "ташкент", "tashkent", "самарканд",
            "samarkand", "бухара", "bukhara", "фергана", "fergana",
            "наманган", "namangan", "андижан", "andijan", "хорезм",
            "khorezm", "навои", "navoi", "каракалпак", "karakalpak",
            "сурхандар", "surkhan", "кашкадар", "kashkadar",
            "jizzakh", "джизак", "syrdarya", "сырдар",
        ]):
            return "uz"
        if any(w in bp for w in [
            "таджик", "tajik", "душанбе", "dushanbe", "худжанд",
            "khujand", "хатлон", "khatlon", "бохтар", "bokhtar",
            "курган-тюбе", "kurgan", "куляб", "kulyab", "kulob",
        ]):
            return "tj"
        if any(w in bp for w in [
            "кирги", "кыргы", "kyrgyz", "бишкек", "bishkek",
            "ош", "osh ", "джалал", "jalal", "нарын", "naryn",
            "каракол", "karakol", "иссык", "issyk",
        ]):
            return "kg"
        if any(w in bp for w in [
            "казах", "kazakh", "алматы", "almaty", "астана", "astana",
            "нур-султан", "nur-sultan", "караганд", "karagand",
            "шымкент", "shymkent", "актобе", "aktobe", "атырау", "atyrau",
        ]):
            return "kz"
        if any(w in bp for w in [
            "азербайджан", "azerba", "баку", "baku", "гянджа",
            "ganja", "сумгаит", "sumgait",
        ]):
            return "az"
        if any(w in bp for w in [
            "армени", "armen", "ереван", "yerevan", "гюмри", "gyumri",
        ]):
            return "am"
        if any(w in bp for w in [
            "молдов", "молдав", "moldov", "кишин", "chisinau",
        ]):
            return "md"
        if any(w in bp for w in [
            "беларус", "белорус", "belarus", "минск", "minsk",
            "гомель", "gomel", "брест", "brest", "гродно", "grodno",
            "витебск", "vitebsk", "могилев", "mogilev",
        ]):
            return "by"
        if any(w in bp for w in [
            "украин", "ukrain", "киев", "kyiv", "kiev", "одесс",
            "odess", "харьков", "kharkiv", "днепр", "dnipr",
            "львов", "lviv", "запорож", "zapori", "донецк", "donetsk",
        ]):
            return "ua"
        if any(w in bp for w in [
            "грузи", "georgi", "тбилис", "tbilisi", "батуми", "batumi",
        ]):
            return "ge"
        if any(w in bp for w in [
            "туркмен", "turkmen", "ашхабад", "ashgabat",
        ]):
            return "tm"

    if passport_number:
        digits = passport_number.replace(" ", "")
        if len(digits) == 10 and digits.isdigit():
            return "ru"

    return "ru"


def get_document_type(passport_number: Optional[str]) -> str:
    """Определяет тип документа."""
    if not passport_number:
        return "PS"

    if len(passport_number.replace(" ", "")) == 10:
        return "PS"

    if passport_number and passport_number[0].isdigit() and len(passport_number) >= 9:
        return "PSP"

    return "NP"


def get_gender_code(gender: Optional[str]) -> str:
    """Преобразует пол в код."""
    if not gender:
        return "m"

    gender_lower = gender.lower()
    if "жен" in gender_lower or "female" in gender_lower or "f" == gender_lower:
        return "f"

    return "m"


def infer_gender(
    middle_name: Optional[str] = None,
    surname: Optional[str] = None,
) -> Optional[str]:
    """Определяет пол по отчеству/патрониму и фамилии."""
    if middle_name:
        mn = middle_name.lower()
        if any(mn.endswith(s) for s in [
            "ович", "евич", "ич",
            "ovich", "evich",
            "ugli", "ogli", "o'g'li",
            "зода", "zoda",
        ]):
            return "male"
        if any(mn.endswith(s) for s in [
            "овна", "евна", "ична",
            "ovna", "evna",
            "qizi", "kizi",
        ]):
            return "female"

    if surname:
        sn = surname.lower()
        if any(sn.endswith(s) for s in ["ova", "eva", "ina", "ова", "ева", "ина"]):
            return "female"
        if any(sn.endswith(s) for s in ["ov", "ev", "in", "ов", "ев", "ин"]):
            return "male"

    return None


def format_date_short(d: Optional[date]) -> str:
    """Форматирует дату в короткий формат DDMMYY."""
    if not d:
        return "000000"
    return d.strftime("%d%m%y")


def format_date_long(d: Optional[date]) -> str:
    """Форматирует дату в длинный формат DDmmmYY."""
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


def _build_format_vars(data: PassportData) -> dict[str, str]:
    """Build all template variables from passport data."""
    country = get_country_code(
        data.birth_place, data.passport_number, data.surname, data.name)
    number = (data.passport_number or "0000000000").replace(" ", "").lower()
    surname = transliterate_to_latin(data.surname).lower()
    name = transliterate_to_latin(data.name).lower()
    gender = get_gender_code(data.gender)
    doc_type = get_document_type(data.passport_number)

    return {
        "country": country,
        "number": number,
        "surname": surname,
        "name": name,
        "gender": gender,
        "doc_type": doc_type,
        "birth_date_long": format_date_long(data.birth_date),
        "birth_date_short": format_date_short(data.birth_date),
        "expiry_long": format_date_long(data.expiry_date),
        "expiry_short": format_date_short(data.expiry_date),
    }


def format_passport_type1(data: PassportData) -> str:
    """Format passport data using template 1 from settings."""
    variables = _build_format_vars(data)
    return settings.format_type1.format(**variables)


def format_passport_type2(data: PassportData) -> str:
    """Format passport data using template 2 from settings."""
    variables = _build_format_vars(data)
    return settings.format_type2.format(**variables)