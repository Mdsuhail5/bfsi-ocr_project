import re
import pandas as pd
from datetime import datetime
from utils.llm import categorize_data


def standardize_date_format(date_str):
    """
    Standardize various date formats to DD-MM-YYYY.

    Args:
        date_str (str): Input date string.

    Returns:
        str: Standardized date string or None if invalid.
    """
    possible_formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d %m %Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d %b %Y", "%b %d, %Y", "%Y-%m-%d"
    ]
    for fmt in possible_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    return None  # Return None if no format matches


def clean_description(description):
    """
    Clean and standardize the description field.

    Args:
        description (str): Input description string.

    Returns:
        str: Cleaned description.
    """
    return re.sub(r"[^a-zA-Z0-9 ]", "", description).strip()


def categorize_and_clean_data(df, document_type):
    """
    Categorize and clean DataFrame records.

    Args:
        df (pd.DataFrame): DataFrame with extracted data.
        document_type (str): Document type for categorization.

    Returns:
        pd.DataFrame: Processed DataFrame with categories.
    """
    df["description"] = df["description"].apply(clean_description)

    # Categorize using LLM
    if "description" in df.columns:
        df["category"] = df["description"].apply(
            lambda x: categorize_data(
                {"description": x}, document_type)["category"]
        )

    # Drop NaN rows
    if document_type == "bank_statement":
        df.dropna(subset=["date", "description", "debit",
                  "credit", "balance"], inplace=True)
    elif document_type == "invoice":
        df.dropna(subset=["product_id", "description",
                  "quantity", "unit_price", "total"], inplace=True)
    elif document_type == "profit_loss":
        df.dropna(subset=["category", "description", "amount"], inplace=True)

    return df


def preprocess_and_postprocess(data, document_type):
    """
    Extract fields and clean data for specific document types.

    Args:
        data (str): Extracted raw text from documents.
        document_type (str): Type of document ('bank_statement', 'invoice', 'profit_loss').

    Returns:
        pd.DataFrame: Processed and categorized DataFrame.
    """
    if document_type == "bank_statement":
        fields = extract_bank_statement_fields(data)
    elif document_type == "invoice":
        fields = extract_invoice_fields(data)
    elif document_type == "profit_loss":
        fields = extract_profit_loss_fields(data)
    else:
        raise ValueError("Unsupported document type")

    df = pd.DataFrame(fields)
    return categorize_and_clean_data(df, document_type)


def extract_bank_statement_fields(text):
    """
    Extract relevant fields from bank statements.
    """
    transactions = []
    lines = text.splitlines()
    for line in lines:
        date_match = re.search(
            r"\b(\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4})\b", line)
        debit_match = re.search(
            r"Debit\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)
        credit_match = re.search(
            r"Credit\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)
        balance_match = re.search(
            r"Balance\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)
        description_match = re.search(
            r"Description\s*:\s*(.+?)\s*(Debit|Credit|Balance|$)", line, re.IGNORECASE)

        if date_match:
            date = standardize_date_format(date_match.group(1))
            debit = debit_match.group(1) if debit_match else None
            credit = credit_match.group(1) if credit_match else None
            balance = balance_match.group(1) if balance_match else None
            description = description_match.group(
                1).strip() if description_match else ""

            transactions.append({
                "date": date,
                "description": description,
                "debit": float(debit) if debit else 0.0,
                "credit": float(credit) if credit else 0.0,
                "balance": float(balance) if balance else 0.0,
            })

    return transactions


def extract_invoice_fields(text):
    """
    Extract fields from invoices.
    """
    products = []
    lines = text.splitlines()
    for line in lines:
        product_id_match = re.search(
            r"Product ID\s*:\s*(\w+)", line, re.IGNORECASE)
        description_match = re.search(
            r"Description\s*:\s*(.+?)\s*(Quantity|Unit Price|Total|$)", line, re.IGNORECASE)
        quantity_match = re.search(
            r"Quantity\s*:\s*(\d+)", line, re.IGNORECASE)
        unit_price_match = re.search(
            r"Unit Price\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)
        total_match = re.search(
            r"Total\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)

        if product_id_match:
            product_id = product_id_match.group(1)
            description = description_match.group(
                1).strip() if description_match else ""
            quantity = int(quantity_match.group(1)) if quantity_match else 0
            unit_price = float(unit_price_match.group(1)
                               ) if unit_price_match else 0.0
            total = float(total_match.group(1)) if total_match else 0.0

            products.append({
                "product_id": product_id,
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "total": total,
            })

    return products


def extract_profit_loss_fields(text):
    """
    Extract fields from profit and loss statements.
    """
    data = []
    lines = text.splitlines()
    for line in lines:
        revenue_match = re.search(
            r"Revenue\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)
        expense_match = re.search(
            r"Expense\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)
        net_profit_match = re.search(
            r"Net Profit\s*:\s*(\d+\.\d{2})", line, re.IGNORECASE)

        if revenue_match or expense_match or net_profit_match:
            entry = {
                "revenue": float(revenue_match.group(1)) if revenue_match else 0.0,
                "expenses": float(expense_match.group(1)) if expense_match else 0.0,
                "net_profit": float(net_profit_match.group(1)) if net_profit_match else 0.0,
            }
            data.append(entry)

    return data
