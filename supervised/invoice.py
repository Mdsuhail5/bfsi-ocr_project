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


def extract_invoice_data(text):
    items = []
    lines = text.splitlines()

    # Regular expressions for field extraction
    product_id_pattern = r"(?:Product|Item)\s*(?:ID|No|#)?\s*[:#]?\s*(\w+)"
    quantity_pattern = r"(?:Qty|Quantity)\s*[:#]?\s*(\d+)"
    price_pattern = r"(?:Price|Rate|Unit Price)\s*[:#]?\s*(\d+,?\d*\.?\d*)"
    amount_pattern = r"(?:Amount|Total)\s*[:#]?\s*(\d+,?\d*\.?\d*)"

    current_item = {}

    for line in lines:
        product_match = re.search(product_id_pattern, line, re.IGNORECASE)
        if product_match:
            if current_item:
                items.append(current_item)
            current_item = {"product_id": product_match.group(1)}

            # Extract description
            desc_start = product_match.end()
            quantity_match = re.search(
                quantity_pattern, line[desc_start:], re.IGNORECASE)
            if quantity_match:
                desc_end = desc_start + quantity_match.start()
                description = line[desc_start:desc_end].strip()
                current_item["description"] = re.sub(
                    r"[^a-zA-Z0-9\s]", "", description)
                current_item["quantity"] = int(quantity_match.group(1))

                # Extract price and total
                price_match = re.search(price_pattern, line, re.IGNORECASE)
                if price_match:
                    current_item["unit_price"] = float(
                        price_match.group(1).replace(',', ''))

                amount_match = re.search(amount_pattern, line, re.IGNORECASE)
                if amount_match:
                    current_item["total"] = float(
                        amount_match.group(1).replace(',', ''))
                elif "unit_price" in current_item and "quantity" in current_item:
                    current_item["total"] = current_item["unit_price"] * \
                        current_item["quantity"]

    if current_item:
        items.append(current_item)

    return items


def process_invoice(file_path):
    raw_text = extract_text_from_file(file_path)
    items = extract_invoice_data(raw_text)
    df = pd.DataFrame(items)

    # Clean and validate data
    required_columns = ['product_id', 'description',
                        'quantity', 'unit_price', 'total']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0 if col in ['quantity', 'unit_price', 'total'] else ''

    # Validate calculations
    df['calculated_total'] = df['quantity'] * df['unit_price']
    df.loc[df['total'].isna(), 'total'] = df['calculated_total']
    df = df.drop('calculated_total', axis=1)

    return df


def save_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    file_path = "path/to/invoice.pdf"  # Replace with actual file path
    output_path = "output/invoice.csv"  # Replace with desired output path

    processed_df = process_invoice(file_path)
    save_to_csv(processed_df, output_path)
    print(f"Processed data saved to {output_path}")
