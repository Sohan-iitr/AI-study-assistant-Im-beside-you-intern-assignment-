# src/components/pdf_processor.py

import fitz  # PyMuPDF library
from typing import Union
from pathlib import Path

def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extracts text content from a given PDF file.

    Args:
        pdf_path (Union[str, Path]): The file path to the PDF document.

    Returns:
        str: A single string containing all the extracted text from the PDF.
             Returns an empty string if the file is not found or is not a PDF.
    """
    # Ensure the path is a Path object for robust handling
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf':
        print(f"Error: The file at {pdf_path} is not a valid PDF file.")
        return ""

    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Initialize an empty list to hold the text of each page
        all_text = []
        
        # Iterate over each page in the PDF
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load the page
            text = page.get_text()         # Extract text from the page
            all_text.append(text)
        
        # Close the document
        doc.close()
        
        # Join all page texts into a single string
        return "\n".join(all_text)

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        return ""

# --- Example of how to test this file directly ---
if __name__ == '__main__':
    # This block will only run when you execute `python pdf_processor.py`
    # It won't run when this file is imported by another script (like app.py)
    
    # Create a dummy PDF in the 'data' folder for testing if it doesn't exist
    # Make sure you have a `data` folder at the root of your project
    test_pdf_path = Path("D:/college/sem5/Heat and mass tranfer/Radiation_I.pdf") # Using relative path from src/components

    # Note: You'll need to create a 'sample.pdf' in your 'data' folder for this to work.
    if test_pdf_path.exists():
        print(f"--- Testing PDF Processor on: {test_pdf_path.name} ---")
        extracted_text = extract_text_from_pdf(test_pdf_path)
        
        if extracted_text:
            print("\nSuccessfully extracted text. First 500 characters:\n")
            print(extracted_text[:500] + "...")
        else:
            print("\nText extraction failed or the document is empty.")
    else:
        print(f"Error: Test file not found at '{test_pdf_path}'.")
        print("Please add a 'sample.pdf' to your 'data' directory to run this test.")