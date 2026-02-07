"""
Passport MRZ reader using Tesseract OCR.

Reads the Machine Readable Zone (MRZ) from Russian passport images.
Returns names in Latin (as they appear in MRZ) instead of converting to Cyrillic.
"""
import imutils
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import cv2
import re
import tempfile
import os


def resize(img_path):
    """Load image, find document contour, crop to document area."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    final_wide = 1200
    r = float(final_wide) / img.shape[1]
    dim = (final_wide, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    big_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c
    if big_contour is None:
        return img
    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)
    nr = np.empty((0, 2), dtype="int32")
    for a in corners:
        for b in a:
            nr = np.vstack([nr, b])
    yarr = [i[0] for i in nr]
    xarr = [i[1] for i in nr]
    x = min(yarr)
    pX = max(yarr)
    y = min(xarr)
    pY = max(xarr)
    photo = img[y:pY, x:pX]
    return photo


def pasp_read(photo):
    """
    Read MRZ from passport image.

    Returns dict with Latin names (as in MRZ) or None on failure.
    Keys: Surname, Name, Mid, Date, Series, Number
    """
    if photo is None:
        return None
    image = photo
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    if maxVal - minVal == 0:
        return None
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="bottom-to-top")[0]
    mrzBox = None
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        percentWidth = w / float(W)
        percentHeight = h / float(H)
        if percentWidth > 0.29 and percentHeight > 0.005:
            mrzBox = (x, y, w, h)
            break
    if mrzBox is None:
        return None
    (x, y, w, h) = mrzBox
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.083)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))
    mrz = image[y:y + h, x:x + w]
    config = " --oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789><"
    mrzText = pytesseract.image_to_string(mrz, lang='eng', config=config)
    mrzText = mrzText.replace(" ", "")
    mrzText = mrzText.split()
    if len(mrzText) < 2:
        return None
    if mrzText[0][0:1] != 'P':
        del mrzText[0]
    if len(mrzText) < 2:
        return None
    el1 = mrzText[0]
    el2 = mrzText[1]
    el1 = el1.replace('1', 'I')
    el2 = el2.replace('O', '0')
    el1 = el1[5:]
    el1 = re.split(r"<<|<|\n", el1)
    el2 = re.split(r"RUS|<", el2)
    el1 = list(filter(None, el1))
    el1 = el1[0:3]
    el2 = list(filter(None, el2))

    # Keep Latin names as-is from MRZ (no Cyrillic conversion)
    surname = el1[0] if len(el1) >= 1 else None
    name = el1[1] if len(el1) >= 2 else None
    otch = el1[2] if len(el1) >= 3 else None

    seria = None
    nomer = None
    data = None
    try:
        if len(el2) >= 3:
            seria = el2[0][0:3] + el2[2][0:1]
            nomer = el2[0][3:9]
            data = el2[1][0:6]
            if int(data[0:1]) > 2:
                data = '19' + data
            else:
                data = '20' + data
            data = data[6:8] + '.' + data[4:6] + '.' + data[0:4]
    except (IndexError, ValueError):
        pass

    pasdata = {
        'Surname': surname,
        'Name': name,
        'Mid': otch,
        'Date': data,
        'Series': seria,
        'Number': nomer,
    }
    return pasdata


def recognize_from_file(image_path):
    """Recognize passport from file path. Returns dict or None."""
    try:
        photo = resize(image_path)
        result = pasp_read(photo)
        return result
    except Exception:
        try:
            photo = cv2.imread(image_path)
            if photo is None:
                return None
            result = pasp_read(photo)
            return result
        except Exception:
            return None


def recognize_from_bytes(image_bytes):
    """Recognize passport from image bytes. Returns dict or None."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, img)
        try:
            result = recognize_from_file(temp_path)
            return result
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    except Exception:
        return None


def catching(image):
    """Legacy function. Returns passport data dict or None."""
    return recognize_from_file(image)