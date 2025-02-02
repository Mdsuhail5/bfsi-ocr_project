# FinSight - Financial Document Analyzer
### Your Financial Documents, Intelligently Decoded

## Overview
FinSight is a powerful financial document analysis tool that leverages computer vision and machine learning to extract, analyze, and visualize data from various financial documents. Built with Streamlit, the application provides an intuitive interface for processing bank statements, invoices, and profit & loss statements.

## 🌟 Features

### Document Analysis
- **Bank Statement Analysis**
  - Transaction extraction from PDF and image formats
  - Detailed transaction categorization
  - Comprehensive financial flow visualization
  - Statistical analysis of income and expenses

- **Invoice Processing**
  - Automated data extraction from invoice images
  - Item-wise breakdown and analysis
  - Tax and discount calculations
  - Vendor and billing information extraction

- **Profit & Loss Statement Analysis**
  - Year-over-year comparison
  - Category-wise financial breakdown
  - Trend analysis and visualization
  - Performance metrics calculation

- **Unsupervised Analysis**
  - BERT-based transaction categorization
  - Intelligent pattern recognition
  - Category-wise financial insights
  - Advanced statistical analysis

### Visualization
- Interactive charts and graphs
- Customizable dashboards
- Real-time data updates
- Export capabilities

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Image Processing**: OpenCV, Tesseract OCR
- **Machine Learning**: BERT, PyTorch
- **Visualization**: Plotly
- **PDF Processing**: PDFPlumber

## 📋 Requirements
```python
streamlit
pandas
numpy
opencv-python
pytesseract
pdfplumber
torch
transformers
plotly
scikit-learn
pillow
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finsight.git
cd finsight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- **Linux**:
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **macOS**:
  ```bash
  brew install tesseract
  ```
- **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## 💻 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload your financial document and select the document type:
   - Bank Statement
   - Invoice
   - Profit and Loss Statement
   - Unsupervised Analysis

4. View the analysis results and visualizations

## 📂 Project Structure
```
finsight/
├── app.py                 # Main Streamlit application
├── visualization.py       # Visualization functions
├── supervised/
│   ├── bank_statement.py # Bank statement processing
│   ├── invoice.py        # Invoice analysis
│   └── pnl.py           # Profit & Loss statement analysis
└── unsupervised/
    └── unsupervised.py  # BERT-based categorization
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for optical character recognition
- [Hugging Face](https://huggingface.co/) for BERT models and transformers
- [Plotly](https://plotly.com/) for interactive visualizations

## 📞 Support
For support, please open an issue in the GitHub repository or contact the maintenance team.