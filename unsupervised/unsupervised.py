import pandas as pd
import pdfplumber
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from supervised.bank_statement import clean_amount, parse_date


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


class BERTCategorizer:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained(
            'bert-base-uncased').to(self.device)

        # Define category descriptions for semantic matching
        self.categories = {
            "Investment": "investment trading stocks mutual funds shares securities trading platform",
            "Transport": "transportation bus metro train taxi cab ride travel fare",
            "Telecom_Utilities": "phone bill mobile recharge internet utilities electricity water gas",
            "Food_Dining": "food restaurant dining cafe takeout delivery groceries supermarket",
            "Education": "education school college university tuition course training institute",
            "Medical_Healthcare": "medical healthcare hospital pharmacy doctor clinic medicine health",
            "Shopping": "shopping retail store purchase merchandise clothes electronics",
            "Entertainment": "entertainment movies theatre sports events recreation leisure",
            "Personal_Transfer": "transfer payment sent received personal individual",
            "Salary": "salary wages income compensation earnings payroll",
            "Insurance": "insurance premium protection coverage policy health life",
            "Tax": "tax government dues levy duty payment official",
            "Rent": "rent lease housing accommodation property payment"
        }

        # Pre-compute category embeddings
        self.category_embeddings = self._get_embeddings(
            list(self.categories.values()))

    def _get_embeddings(self, texts):
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings.append(
                    outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(embeddings)

    def _get_description_embedding(self, description):
        with torch.no_grad():
            inputs = self.tokenizer(
                description, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def categorize(self, description):
        # Get embedding for the transaction description
        desc_embedding = self._get_description_embedding(description)

        # Calculate similarity with all categories
        similarities = cosine_similarity(
            desc_embedding, self.category_embeddings)

        # Get the most similar category
        category_idx = similarities.argmax()
        category = list(self.categories.keys())[category_idx]

        return category


class UnsupervisedAnalyzer:
    def __init__(self):
        self.categorizer = BERTCategorizer()

    def analyze_transactions(self, file_path, is_pdf=True):
        if is_pdf:
            df = extract_transactions_from_pdf(file_path)
        else:
            df = pd.read_csv(file_path)

        # Apply BERT-based categorization
        df['Category'] = df['Description'].apply(self.categorizer.categorize)

        # Calculate category-wise statistics
        category_stats = df.groupby('Category').agg({
            'Debit': 'sum',
            'Credit': 'sum'
        }).reset_index()

        category_stats['Net'] = category_stats['Credit'] - \
            category_stats['Debit']

        # Add confidence scores if needed
        # df['Category_Confidence'] = ...

        return df, category_stats


if __name__ == "__main__":
    analyzer = UnsupervisedAnalyzer()
    df, stats = analyzer.analyze_transactions('bk1-3.pdf')
    df.to_csv('categorized_transactions_bert.csv', index=False)
    print("Transactions categorized using BERT and saved to 'categorized_transactions_bert.csv'")
