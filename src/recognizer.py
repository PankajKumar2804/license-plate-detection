"""
OCR module for license plate text recognition
"""

import pytesseract
import cv2
from typing import Optional


class PlateRecognizer:
    """Handles OCR for license plate text extraction"""
    
    def __init__(self, language: str = 'eng'):
        """
        Initialize recognizer
        
        Args:
            language: Tesseract language code
        """
        self.language = language
        
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from license plate image
        
        Args:
            image_path: Path to license plate image
            
        Returns:
            Extracted text
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocess for better OCR
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
        
        # Extract text
        text = pytesseract.image_to_string(thresh)
        return text.strip()
    
    def extract_text_from_region(self, image, bbox: list) -> str:
        """
        Extract text from a bounding box region
        
        Args:
            image: OpenCV image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Extracted text
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        plate_region = image[y1:y2, x1:x2]
        
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
        
        text = pytesseract.image_to_string(thresh)
        return text.strip()


if __name__ == "__main__":
    recognizer = PlateRecognizer()
    print("Recognizer initialized successfully")
