import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
from datetime import datetime
from pdf2image import convert_from_path
import os


def standardize_date_format(date_str):
    possible_formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d %m %Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d %b %Y", "%b %d, %Y", "%Y-%m-%d"
    ]
    for fmt in possible_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    return None


def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(binary_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced


def extract_text_from_file(file_path):
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path, dpi=300)
        text = []
        for page in pages:
            open_cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            processed_image = preprocess_image(open_cv_image)
            text.append(pytesseract.image_to_string(processed_image))
        return '\n'.join(text)
    else:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image file: {file_path}")
        processed_image = preprocess_image(image)
        return pytesseract.image_to_string(processed_image)


def extract_bank_statement_data(text):
    transactions = []
    lines = text.splitlines()

    date_pattern = r"\b(\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4})\b"
    amount_pattern = r"(\d+,?\d*\.?\d*)"

    current_transaction = {}

    for line in lines:
        date_match = re.search(date_pattern, line)
        if date_match:
            if current_transaction:
                transactions.append(current_transaction)
            current_transaction = {
                "date": standardize_date_format(date_match.group(1))}

            desc_start = date_match.end()
            amounts = re.finditer(amount_pattern, line[desc_start:])
            amounts = list(amounts)
            if amounts:
                desc_end = desc_start + amounts[0].start()
                description = line[desc_start:desc_end].strip()
                current_transaction["description"] = re.sub(
                    r"[^a-zA-Z0-9\s]", "", description)

                if len(amounts) >= 2:
                    amount1 = float(amounts[0].group(1).replace(',', ''))
                    amount2 = float(amounts[1].group(1).replace(',', ''))

                    if "DR" in line or "debit" in line.lower():
                        current_transaction["debit"] = amount1
                        current_transaction["credit"] = 0
                    elif "CR" in line or "credit" in line.lower():
                        current_transaction["debit"] = 0
                        current_transaction["credit"] = amount1
                    else:
                        current_transaction["debit"] = amount1 if amount1 > 0 else 0
                        current_transaction["credit"] = amount2 if amount2 > 0 else 0

                    if len(amounts) >= 3:
                        current_transaction["balance"] = float(
                            amounts[2].group(1).replace(',', ''))

    if current_transaction:
        transactions.append(current_transaction)

    return transactions


def process_bank_statement(file_path):
    raw_text = extract_text_from_file(file_path)
    transactions = extract_bank_statement_data(raw_text)
    df = pd.DataFrame(transactions)

    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    df['date'] = df['date'].apply(lambda x: x.strftime(
        '%Y-%m-%d %H:%M:%S') if pd.notnull(x) else x)

    required_columns = ['date', 'description', 'debit', 'credit', 'balance']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0 if col in ['debit', 'credit', 'balance'] else ''

    visualization_df = pd.DataFrame({
        'category': df['description'],
        'amount': df['debit'] + df['credit']
    })

    return df, visualization_df
