"""
Unit tests for license plate detection
"""

import unittest
import os
from src.detector import LicensePlateDetector
from src.recognizer import PlateRecognizer


class TestLicensePlateDetector(unittest.TestCase):
    """Test cases for detector"""
    
    @classmethod
    def setUpClass(cls):
        cls.detector = LicensePlateDetector(confidence=0.5)
        cls.recognizer = PlateRecognizer()
    
    def test_detector_initialization(self):
        """Test detector initializes correctly"""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.confidence, 0.5)
    
    def test_recognizer_initialization(self):
        """Test recognizer initializes correctly"""
        self.assertIsNone(self.recognizer.extract_text('test.jpg') is None 
                         or isinstance(self.recognizer.extract_text('test.jpg'), str))


if __name__ == '__main__':
    unittest.main()
