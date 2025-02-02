import pandas as pd
import pdfplumber
import re
from datetime import datetime
import pytesseract
from PIL import Image
import cv2
import numpy as np


def clean_amount(amount_str):
    if not amount_str:
        return 0.0
    try:
        return float(amount_str.replace(',', ''))
    except ValueError:
        return 0.0


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%m,%d,%Y').strftime('%Y-%m-%d')
    except:
        return date_str


def extract_transactions_from_pdf(pdf_path):
    transactions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            lines = text.split('\n')

            for line in lines:
                if any(x in line.lower() for x in ['date from', 'transactions for', 'last n transactions']):
                    continue

                date_match = re.search(r'(\d{2},\d{2},\d{4})', line)
                if date_match:
                    parts = line.split()

                    date = date_match.group(1)

                    desc_start = line.index(date) + len(date)
                    amounts = re.findall(
                        r'(?:^|\s)(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', line[desc_start:])
                    desc_end = line.rindex(
                        amounts[-1]) if amounts else len(line)
                    description = line[desc_start:desc_end].strip()

                    if len(amounts) >= 2:
                        balance = amounts[-1]
                        amount = amounts[-2]
                        prev_balance = float(balance.replace(',', ''))
                        curr_amount = float(amount.replace(',', ''))

                        if prev_balance < curr_amount:
                            debit, credit = amount, '0.00'
                        else:
                            debit, credit = '0.00', amount

                        transaction = {
                            'Date': parse_date(date),
                            'Description': description,
                            'Debit': debit,
                            'Credit': credit,
                            'Balance': balance
                        }
                        transactions.append(transaction)

    df = pd.DataFrame(transactions, columns=[
                      'Date', 'Description', 'Debit', 'Credit', 'Balance'])

    for col in ['Debit', 'Credit', 'Balance']:
        if col in df.columns:
            df[col] = df[col].apply(clean_amount)

    return df


def preprocess_image(image):
    """Enhanced image preprocessing for better OCR results"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(contrast)

    # Adaptive thresholding
    binary = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return opening


def extract_transactions_from_text(text):
    """Enhanced transaction extraction from text"""
    lines = text.split('\n')
    transactions = []

    for line in lines:
        # Match various date formats
        if re.search(r'\d{2}[-/][A-Za-z]{3}|\d{2}[-/]\d{2}[-/]\d{4}', line):
            parts = re.split(r'\s{2,}', line.strip())

            if len(parts) >= 3:
                # Extract all numbers from the line
                numbers = [n for n in parts if re.match(
                    r'^[-]?\d+\.?\d*$', n.replace(',', ''))]

                if len(numbers) >= 2:
                    date = parts[0]
                    desc_parts = parts[1:-len(numbers)]
                    description = ' '.join(
                        desc_parts) if desc_parts else 'Unknown'

                    # Handle different transaction formats
                    if len(numbers) >= 3:
                        debit = clean_amount(numbers[-3])
                        credit = clean_amount(numbers[-2])
                    else:
                        amount = clean_amount(numbers[-2])
                        debit = amount if amount > 0 else 0
                        credit = -amount if amount < 0 else 0

                    balance = clean_amount(numbers[-1])

                    transaction = {
                        'Date': date,
                        'Description': description,
                        'Debit': debit,
                        'Credit': credit,
                        'Balance': balance
                    }
                    transactions.append(transaction)

    return pd.DataFrame(transactions)


def extract_transactions_from_image(image_path):
    """Extract transactions from bank statement image with enhanced processing"""
    # Read and validate image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Scale up image for better OCR
    scale_factor = 2
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # Preprocess image
    processed = preprocess_image(image)

    # Configure and perform OCR
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(processed, config=custom_config)

    # Extract and process transactions
    df = extract_transactions_from_text(text)

    # Clean up the data
    for col in ['Debit', 'Credit', 'Balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by date if possible
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        df = df.sort_values('Date')
    except:
        pass  # Keep original order if date parsing fails

    return df
