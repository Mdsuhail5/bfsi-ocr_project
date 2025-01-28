import os
from pathlib import Path


def handle_file_upload(uploaded_files, upload_dir="uploads"):
    """
    Handle file uploads by saving them to a specified directory.

    Args:
        uploaded_files (list): List of uploaded files from Streamlit.
        upload_dir (str): Directory to save uploaded files.

    Returns:
        list: List of saved file paths.
    """
    # Ensure the upload directory exists
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    saved_file_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)

        # Write the uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_file_paths.append(file_path)

    return saved_file_paths
