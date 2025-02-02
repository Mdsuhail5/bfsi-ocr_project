import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
from datetime import datetime
import re


class InvoiceAnalyzer:
    def __init__(self):
        self.processed_invoices = []
        self.total_processed = 0

    def preprocess_image(self, image_path):
        """Preprocess the image for better OCR accuracy"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Apply dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.dilate(gray, kernel, iterations=1)

        temp_path = os.path.join(os.path.dirname(
            image_path), "temp_processed_image.png")
        cv2.imwrite(temp_path, gray)

        return temp_path

    def extract_data(self, image_path):
        """Extract data from invoice image"""
        processed_image_path = self.preprocess_image(image_path)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(Image.open(
            processed_image_path), config=custom_config)

        # Parse the extracted text
        lines = text.split('\n')

        # Extract invoice details
        invoice_data = {
            'items': [],
            'summary': {},
            'billing_info': {},
            'metadata': {
                'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_file': os.path.basename(image_path)
            }
        }

        # Extract billing information
        billing_info_start = False
        for i, line in enumerate(lines):
            if "Bill to:" in line:
                billing_info_start = True
                continue
            if billing_info_start:
                if line.strip() == "":
                    break
                if "Company name" in line:
                    invoice_data['billing_info']['customer_name'] = lines[i - 1].strip()
                    invoice_data['billing_info']['company_name'] = line.strip()
                elif "Company Adrress and" in line:
                    invoice_data['billing_info']['address'] = lines[i - 1].strip()
                    invoice_data['billing_info']['location'] = lines[i + 1].strip()

        # Extract items
        items_start = False
        for line in lines:
            if "No/Code" in line and "Product Description" in line and "Qty" in line and "Price" in line and "Total" in line:
                items_start = True
                continue
            if items_start:
                if "Sub Total" in line:
                    break
                parts = line.split()
                if len(parts) >= 5:
                    item = {
                        'no_code': parts[0],
                        'description': ' '.join(parts[1:-3]),
                        'qty': parts[-3],
                        'price': parts[-2],
                        'total': parts[-1]
                    }
                    invoice_data['items'].append(item)

        # Extract summary information
        summary_items = ['Sub Total', 'Tax/VAT', 'Discount', 'Grand Total']
        for line in lines:
            for item in summary_items:
                if item in line:
                    amount = re.findall(r'\d+', line)
                    if amount:
                        invoice_data['summary'][item.lower().replace(
                            '/', '_')] = float(amount[-1])

        self.processed_invoices.append(invoice_data)
        self.total_processed += 1

        return invoice_data

    def analyze_invoice(self, invoice_data):
        """Analyze invoice data and generate insights"""
        if not invoice_data['items']:
            return {
                'total_items': 0,
                'average_item_price': 0,
                'highest_priced_item': {'description': 'N/A', 'price': 'N/A'},
                'lowest_priced_item': {'description': 'N/A', 'price': 'N/A'},
                'tax_percentage': 0,
                'discount_percentage': 0
            }

        analysis = {
            'total_items': len(invoice_data['items']),
            'average_item_price': np.mean([float(item['price']) for item in invoice_data['items']]),
            'highest_priced_item': max(invoice_data['items'], key=lambda x: float(x['price'])),
            'lowest_priced_item': min(invoice_data['items'], key=lambda x: float(x['price'])),
            'tax_percentage': (invoice_data['summary'].get('tax_vat', 0) / invoice_data['summary'].get('sub_total', 1)) * 100,
            'discount_percentage': (invoice_data['summary'].get('discount', 0) / invoice_data['summary'].get('sub_total', 1)) * 100
        }
        return analysis
