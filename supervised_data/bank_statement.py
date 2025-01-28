import pytesseract
import cv2
import os
import numpy as np
import tempfile
import pdfplumber
from pdf2image import convert_from_path


def process_bank_statement(file):
    """Process a bank statement file to extract and categorize data.

    Args:
        file (UploadedFile): The uploaded file object.

    Returns:
        str: Extracted text from the file.
    """
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == ".pdf":
        return extract_text_from_pdf(file)
    elif file_extension in (".jpg", ".jpeg", ".png"):
        return extract_text_from_image(file)
    else:
        raise ValueError(
            "Unsupported file type. Only PDFs, JPG, JPEG, and PNG are allowed."
        )


def extract_text_from_image(file):
    """Extract text from an image file using Tesseract OCR.

    Args:
        file (UploadedFile): The uploaded image file.

    Returns:
        str: Extracted text.
    """
    # Convert bytes to numpy array
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR using Tesseract
    text = pytesseract.image_to_string(binary_image)

    return text


def extract_text_from_pdf(file):
    """Extract text from a PDF file using pdfplumber and Tesseract OCR.

    Args:
        file (UploadedFile): The uploaded PDF file.

    Returns:
        str: Extracted text.
    """
    text = ""
    # Save temporarily and convert to images
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(file.getvalue())
        pages = convert_from_path(tmp_pdf.name, dpi=300)
        for page in pages:
            text += pytesseract.image_to_string(page)
        os.unlink(tmp_pdf.name)
    return text
