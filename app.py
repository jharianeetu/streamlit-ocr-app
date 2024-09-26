import streamlit as st
import pytesseract
import cv2
import re
import os

def preprocess_image(image):
    """
    Preprocess the image for better OCR accuracy.
    - Convert to grayscale
    - Apply thresholding
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Applying a binary threshold to make the text stand out
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresholded

def perform_ocr(image_path):
    """
    Performs OCR on an image using PyTesseract.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Extracted text from the image.
    """
    # Load the image
    img = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Performing OCR with both Hindi and English languages
    extracted_text = pytesseract.image_to_string(preprocessed_img, lang='hin+eng')  

    return extracted_text

def highlight_matches(text, keywords):
    """
    Highlights keywords within the extracted text.
    Args:
        text (str): Extracted text from the image.
        keywords (list): List of keywords to search for.
    Returns:
        str: Extracted text with highlighted keywords.
    """
    for keyword in keywords:
        pattern = rf"\b{keyword}\b"  # Use word boundaries for accurate matching
        text = re.sub(pattern, f"<mark>{keyword}</mark>", text, flags=re.IGNORECASE)  # Highlight using HTML mark tag, case-insensitive
    return text

def main():
    """
    Main function for the Streamlit app.
    """
    st.title("OCR-Based Text Extraction with Keyword Search")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image")

    if uploaded_image is not None:
        # Save the uploaded image to a temporary file
        temp_filepath = f"temp_image.{uploaded_image.name.split('.')[-1]}"
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_image.read())

        # Process image using OCR
        extracted_text = perform_ocr(temp_filepath)

        # Keyword search input
        keywords = st.text_input("Enter keywords (comma-separated):")
        keyword_list = keywords.split(",") if keywords else []  # Split comma-separated keywords

        # Display extracted text with highlighted matches
        if keyword_list:
            highlighted_text = highlight_matches(extracted_text, keyword_list)
            st.markdown(highlighted_text, unsafe_allow_html=True)  # Allow HTML for highlighting
        else:
            st.text_area("Extracted Text:", extracted_text)

        # Delete temporary file
        os.remove(temp_filepath)

if __name__ == "__main__":
    main()
