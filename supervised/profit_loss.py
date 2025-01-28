import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
from pdf2image import convert_from_path


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


def extract_profit_loss_data(text):
    entries = []
    lines = text.splitlines()

    # Regular expressions for field extraction
    category_pattern = r"^(revenue|income|sales|expense|cost|profit|loss)"
    amount_pattern = r"[£$€]?\s*(\d+,?\d*\.?\d*)"
    date_pattern = r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{4})\b"

    current_entry = {}
    current_category = None

    for line in lines:
        line = line.lower().strip()

        # Skip empty lines
        if not line:
            continue

        category_match = re.match(category_pattern, line, re.IGNORECASE)
        if category_match:
            if current_entry:
                entries.append(current_entry)
                current_entry = {}

            current_category = category_match.group(1)
            if 'revenue' in current_category or 'income' in current_category or 'sales' in current_category:
                current_category = 'revenue'
            elif 'expense' in current_category or 'cost' in current_category:
                current_category = 'expenses'
            elif 'profit' in current_category or 'loss' in current_category:
                current_category = 'net_profit'

        amount_match = re.search(amount_pattern, line)
        if amount_match and current_category:
            amount = float(amount_match.group(1).replace(',', ''))
            current_entry[current_category] = amount

        date_match = re.search(date_pattern, line)
        if date_match:
            current_entry['date'] = date_match.group(1)

        # Extract description (text between category and amount)
        if category_match and amount_match:
            desc_start = category_match.end()
            desc_end = amount_match.start()
            description = line[desc_start:desc_end].strip()
            if description:
                current_entry['description'] = re.sub(
                    r"[^a-zA-Z0-9\s]", "", description)

    if current_entry:
        entries.append(current_entry)

    return entries


def process_profit_loss(file_path):
    raw_text = extract_text_from_file(file_path)
    entries = extract_profit_loss_data(raw_text)
    df = pd.DataFrame(entries)

    # Clean and validate data
    required_columns = ['date', 'description',
                        'revenue', 'expenses', 'net_profit']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0 if col in ['revenue', 'expenses', 'net_profit'] else ''

    # Calculate net profit if missing
    df.loc[df['net_profit'] == 0, 'net_profit'] = df['revenue'] - df['expenses']

    # Sort by date if present
    if 'date' in df.columns and not df['date'].isna().all():
        df['date'] = pd.to_datetime(
            df['date'], format='%Y-%m-%d', errors='coerce')
        df = df.sort_values('date')

    return df


def save_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    file_path = "path/to/profit_loss.pdf"  # Replace with actual file path
    output_path = "output/profit_loss.csv"  # Replace with desired output path

    processed_df = process_profit_loss(file_path)
    save_to_csv(processed_df, output_path)
    print(f"Processed data saved to {output_path}")
