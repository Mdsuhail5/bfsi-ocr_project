# utils/llm.py
import os
from pathlib import Path


from transformers import pipeline

# Initialize the BERT model for text classification
# You can replace this with a local model if needed.


def initialize_model():
    """
    Initialize the BERT model for categorization tasks.
    """
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def categorize_with_bert(description, candidate_labels):
    """
    Use BERT to categorize a given description into predefined categories.

    Args:
        description (str): The transaction or invoice description to classify.
        candidate_labels (list): A list of predefined categories.

    Returns:
        str: The most likely category based on BERT's output.
    """
    classifier = initialize_model()
    result = classifier(description, candidate_labels)
    return result["labels"][0]  # Return the highest-scoring label


def categorize_data(data, document_type):
    """
    Categorize data based on its type using BERT or fallback methods.

    Args:
        data (dict): The extracted and preprocessed data.
        document_type (str): The type of document (e.g., 'bank_statement', 'invoice').

    Returns:
        dict: The categorized data with an added 'category' field.
    """
    # Define candidate labels based on the document type
    if document_type == "bank_statement":
        candidate_labels = [
            "Food & Dining", "Travel", "Shopping", "Utilities",
            "Healthcare", "Entertainment", "Salary", "Rent"
        ]
    elif document_type == "invoice":
        candidate_labels = ["Office Supplies",
                            "Services", "Utilities", "Technology"]
    elif document_type == "profit_loss":
        candidate_labels = ["Revenue", "Expenses", "Net Income"]
    else:
        raise ValueError("Unsupported document type")

    # Attempt categorization with BERT
    description = data.get("description", "")
    if description:
        data["category"] = categorize_with_bert(description, candidate_labels)
    else:
        data["category"] = "Uncategorized"

    return data
