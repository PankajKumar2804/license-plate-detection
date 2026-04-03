# License Plate Detection

Real-time license plate detection and recognition system using YOLOv8 and OCR.

## Features
- Real-time detection from images and video
- YOLO v8 object detection
- Tesseract OCR for text extraction
- REST API support
- Multi-format plate support

## Installation
\\\ash
pip install -r requirements.txt
\\\

## Usage
\\\python
from src.detector import LicensePlateDetector
detector = LicensePlateDetector()
results = detector.detect('image.jpg')
\\\

## Tech Stack
- Python 3.9+
- YOLOv8, OpenCV
- FastAPI, Tesseract OCR

## Author
Pankaj Kumar
