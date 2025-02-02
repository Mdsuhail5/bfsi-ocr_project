import streamlit as st
import os
import pandas as pd
from supervised.bank_statement import extract_transactions_from_pdf, extract_transactions_from_image
from supervised.invoice import InvoiceAnalyzer
from supervised.pnl import ProfitLossExtractor
import visualization as viz
from unsupervised.unsupervised import UnsupervisedAnalyzer
from streamlit.components.v1 import html
import base64


def set_page_config():
    st.set_page_config(
        page_title="FinSight - Financial Document Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def load_css():
    st.markdown("""
        <style>
        .main {
            padding-top: 0rem;
        }
        .title-text {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle-text {
            font-size: 1.2rem;
            color: #FFDDC1;
        }
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            height: 3rem;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


def display_header():
    st.markdown('<p class="title-text">FinSight</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Your Financial Documents, Intelligently Decoded</p>',
                unsafe_allow_html=True)


def file_uploader_section():
    st.sidebar.title("📤 Upload Document")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Supported formats: PDF, PNG, JPG, JPEG"
    )

    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.sidebar.success("File uploaded successfully!")
        st.sidebar.json(file_details)

    return uploaded_file


def document_type_selector():
    return st.sidebar.selectbox(
        "📄 Select Document Type",
        ["Bank Statement", "Invoice",
            "Profit and Loss Statement", "Unsupervised Analysis"],
        help="Choose the type of document you're analyzing"
    )


def process_document(uploaded_file, file_type_option):
    if uploaded_file is None:
        st.info("👆 Upload a document to get started!")
        return

    try:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if file_type_option == "Bank Statement":
                process_bank_statement(uploaded_file)
            elif file_type_option == "Invoice":
                process_invoice(uploaded_file)
            elif file_type_option == "Profit and Loss Statement":
                process_pnl(uploaded_file)
            elif file_type_option == "Unsupervised Analysis":
                process_unsupervised(uploaded_file)

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

    finally:
        # Cleanup
        if os.path.exists(uploaded_file.name):
            os.remove(uploaded_file.name)


def process_bank_statement(uploaded_file):
    if uploaded_file.type == "application/pdf":
        df = extract_transactions_from_pdf(uploaded_file.name)
    else:
        df = extract_transactions_from_image(uploaded_file.name)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("📊 Extracted Transactions")
    with col2:
        st.download_button(
            "⬇️ Download Data",
            df.to_csv(index=False),
            "transactions.csv",
            "text/csv"
        )

    st.dataframe(df.style.format({
        'Debit': '{:,.2f}',
        'Credit': '{:,.2f}',
        'Balance': '{:,.2f}'
    }))

    viz.display_visualizations(df, document_type="Bank Statement")


def process_invoice(uploaded_file):
    analyzer = InvoiceAnalyzer()
    invoice_data = analyzer.extract_data(uploaded_file.name)
    analysis = analyzer.analyze_invoice(invoice_data)

    # Display results in tabs
    tab1, tab2 = st.tabs(["📊 Analysis", "📋 Details"])

    with tab1:
        analysis_df = pd.DataFrame([analysis])
        st.dataframe(analysis_df)

    with tab2:
        if 'items' in invoice_data:
            st.write("📝 Invoice Items")
            items_df = pd.DataFrame(invoice_data['items'])
            st.dataframe(items_df)

    viz.display_visualizations(invoice_data, analysis, document_type="Invoice")


def process_pnl(uploaded_file):
    extractor = ProfitLossExtractor()
    pnl_data = extractor.extract_pnl_data(uploaded_file.name)

    st.write("📈 Profit and Loss Analysis")
    st.dataframe(pnl_data)

    viz.display_visualizations(pnl_data, document_type="PnL")


def process_unsupervised(uploaded_file):
    analyzer = UnsupervisedAnalyzer()
    is_pdf = uploaded_file.type == "application/pdf"
    df, category_stats = analyzer.analyze_transactions(
        uploaded_file.name, is_pdf=is_pdf)

    st.write("🔍 Unsupervised Analysis Results")
    st.dataframe(df)

    viz.display_unsupervised_visualizations(df, category_stats)


def main():
    set_page_config()
    load_css()
    display_header()

    uploaded_file = file_uploader_section()
    file_type_option = document_type_selector()

    # Add help section in sidebar
    with st.sidebar.expander("ℹ️ Help"):
        st.markdown("""
        **How to use FinSight:**
        1. Upload your financial document
        2. Select the document type
        3. View the analysis and visualizations
        
        **Supported Documents:**
        - Bank Statements
        - Invoices
        - Profit & Loss Statements
        """)

    process_document(uploaded_file, file_type_option)


if __name__ == "__main__":
    main()
