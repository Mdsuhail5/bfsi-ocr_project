import streamlit as st
import pandas as pd
import tempfile
import os

from supervised.bank_statement import process_bank_statement
from supervised.invoice import process_invoice
from supervised.profit_loss import process_profit_loss
from utils.database import initialize_database, save_to_db
from visualization import visualize_data

st.title("BFSI OCR - Financial Document Analyzer")

# Initialize the database
initialize_database()

menu = ["Bank Statements", "Invoices",
        "Profit & Loss Statements", "Unsupervised Data"]
choice = st.sidebar.selectbox("Select Document Type", menu)


def process_uploaded_file(uploaded_file, processor_func):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.flush()
        tmp_file.seek(0)
        processed_df, visualization_df = processor_func(tmp_file.name)
    tmp_file.close()  # Close the file explicitly before deletion
    os.unlink(tmp_file.name)
    return processed_df, visualization_df


if choice == "Bank Statements":
    st.header("Upload Bank Statements")
    uploaded_files = st.file_uploader(
        "Upload your bank statement(s) (PNG, JPG, JPEG, PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Processing file: {file.name}")
            processed_df, visualization_df = process_uploaded_file(
                file, process_bank_statement)
            save_to_db("bank_statements", processed_df.to_dict("records"))
            st.success("Data saved to database")
            st.write("Data Visualization:")
            visualize_data(visualization_df, chart_type="bar")

elif choice == "Invoices":
    st.header("Upload Invoices")
    uploaded_files = st.file_uploader(
        "Upload your invoice(s) (PNG, JPG, JPEG, PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Processing file: {file.name}")
            processed_df, visualization_df = process_uploaded_file(
                file, process_invoice)
            save_to_db("invoices", processed_df.to_dict("records"))
            st.success("Data saved to database")
            st.write("Data Visualization:")
            visualize_data(visualization_df, chart_type="pie")

elif choice == "Profit & Loss Statements":
    st.header("Upload Profit & Loss Statements")
    uploaded_files = st.file_uploader(
        "Upload your profit & loss statement(s) (PNG, JPG, JPEG, PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Processing file: {file.name}")
            processed_df, visualization_df = process_uploaded_file(
                file, process_profit_loss)
            save_to_db("profit_loss", processed_df.to_dict("records"))
            st.success("Data saved to database")
            st.write("Data Visualization:")
            visualize_data(visualization_df, chart_type="line")
