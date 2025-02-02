import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
import os


class ProfitLossExtractor:
    def __init__(self):
        self.known_sections = {
            'revenue': ['Revenue', 'Total Revenue & Gains'],
            'expenses': ['Expenses', 'Total Expenses'],
            'income': ['Income before tax', 'Income tax expense', 'Net Profit (Loss)']
        }

    def preprocess_image(self, image):
        """Preprocess the image for better OCR accuracy"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(contrast)
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def extract_amount(self, text):
        """Extract numerical amount from text"""
        if not isinstance(text, str):
            return None
        cleaned = text.replace('$', '').replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return None

    def extract_pnl_data(self, image_path):
        """Extract profit and loss data from an image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        scale_factor = 2
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        processed = self.preprocess_image(image)

        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(processed, config=custom_config)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        data = []
        current_section = 'Unknown'

        for line in lines:
            if any(section in line for section_list in self.known_sections.values() for section in section_list):
                current_section = line.strip()
                continue

            parts = [p.strip()
                     for p in re.split(r'\s{2,}|\t+', line) if p.strip()]

            if len(parts) >= 3:
                category = parts[0]
                amount_2021 = self.extract_amount(parts[1])
                amount_2022 = self.extract_amount(parts[2])

                if amount_2021 is not None and amount_2022 is not None:
                    data.append({
                        'Section': current_section,
                        'Category': category,
                        '2021': amount_2021,
                        '2022': amount_2022,
                        'YoY_Change': ((amount_2022 - amount_2021) / amount_2021 * 100) if amount_2021 != 0 else 0
                    })

        df = pd.DataFrame(data)

        # Format currency values
        for year in ['2021', '2022']:
            df[year] = df[year].apply(
                lambda x: f"${x:,.2f}" if pd.notnull(x) else '')

        df['YoY_Change'] = df['YoY_Change'].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) else '')

        return df
