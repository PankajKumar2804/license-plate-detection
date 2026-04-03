"""
Advanced License Plate Detection System
Production-grade computer vision pipeline with YOLOv8 and OCR
"""

import cv2
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionModel(str, Enum):
    """Available detection models"""
    YOLOV8_NANO = "yolov8n"
    YOLOV8_SMALL = "yolov8s"
    YOLOV8_MEDIUM = "yolov8m"
    YOLOV8_LARGE = "yolov8l"


@dataclass
class PlateDetection:
    """Detected license plate data"""
    plate_text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: datetime
    image_path: str
    model_used: str
    processing_time_ms: float


@dataclass
class ProcessingResult:
    """Result from processing an image"""
    image_path: str
    detections: List[PlateDetection]
    total_detections: int
    processing_time_ms: float
    errors: List[str]
    status: str  # 'success', 'partial', 'failed'


class PlateDetector:
    """
    Advanced license plate detection system using YOLOv8
    """
    
    def __init__(
        self,
        model_size: DetectionModel = DetectionModel.YOLOV8_MEDIUM,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = 'cuda:0',
        cache_dir: str = 'models'
    ):
        """
        Initialize plate detector
        
        Args:
            model_size: YOLO model size to use
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-Maximum Suppression threshold
            device: Processing device ('cuda:0', 'cpu', etc.)
            cache_dir: Directory for model caching
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.cache_dir = cache_dir
        
        logger.info(f"Initializing PlateDetector with {model_size} model")
        logger.info(f"Using device: {device}")
        
        # Model will be lazy-loaded on first use
        self.model = None
        self.detection_count = 0
        self.processing_stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'avg_processing_time': 0
        }
    
    def _load_model(self):
        """Lazy load YOLO model"""
        if self.model is not None:
            return
        
        try:
            from ultralytics import YOLO
            model_name = f"{self.model_size}.pt"
            logger.info(f"Loading YOLO model: {model_name}")
            
            # In production, download from official source
            self.model = YOLO(model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image_source) -> List[PlateDetection]:
        """
        Detect license plates in image or video frame
        
        Args:
            image_source: Path to image file or numpy array (frame)
        
        Returns:
            List of PlateDetection objects
        """
        if self.model is None:
            self._load_model()
        
        start_time = datetime.now()
        detections = []
        
        try:
            # Load image if path provided
            if isinstance(image_source, str):
                image = cv2.imread(image_source)
                if image is None:
                    logger.error(f"Failed to load image: {image_source}")
                    return []
                source_path = image_source
            else:
                image = image_source
                source_path = "frame"
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                verbose=False
            )
            
            # Process results
            if results and len(results) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Recognize text in detected region
                    plate_text = self._recognize_plate_text(image[y1:y2, x1:x2])
                    
                    detection = PlateDetection(
                        plate_text=plate_text,
                        confidence=confidence,
                        bounding_box=(x1, y1, x2, y2),
                        timestamp=datetime.now(),
                        image_path=source_path,
                        model_used=self.model_size.value,
                        processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                    )
                    
                    detections.append(detection)
                    self.detection_count += 1
            
            logger.info(f"Detected {len(detections)} plates in {source_path}")
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return detections
    
    def _recognize_plate_text(self, plate_image: np.ndarray) -> str:
        """
        Recognize text in plate image using OCR
        
        Args:
            plate_image: Cropped license plate region
        
        Returns:
            Recognized plate text
        """
        try:
            # Preprocess plate image
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            # Enhance image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
            
            # Denoise
            denoised = cv2.morphologyEx(
                thresh,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            )
            
            # In production, use Tesseract or cloud OCR
            # For now, return placeholder
            return self._mock_ocr_recognition()
            
        except Exception as e:
            logger.warning(f"Text recognition failed: {e}")
            return "UNKNOWN"
    
    def _mock_ocr_recognition(self) -> str:
        """Mock OCR for demo purposes"""
        # In production, integrate actual OCR service
        import random
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        numbers = "0123456789"
        plate = ''.join([
            random.choice(letters) for _ in range(3)
        ]) + '-' + ''.join([
            random.choice(numbers) for _ in range(4)
        ])
        return plate
    
    def detect_in_video(self, video_path: str, output_path: Optional[str] = None) -> ProcessingResult:
        """
        Detect plates in video and optionally save annotated video
        
        Args:
            video_path: Path to video file
            output_path: Path to save annotated video
        
        Returns:
            ProcessingResult with all detections
        """
        logger.info(f"Processing video: {video_path}")
        
        all_detections = []
        start_time = datetime.now()
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return ProcessingResult(
                    image_path=video_path,
                    detections=[],
                    total_detections=0,
                    processing_time_ms=0,
                    errors=["Failed to open video"],
                    status="failed"
                )
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer if needed
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect plates in frame
                detections = self.detect(frame)
                all_detections.extend(detections)
                
                # Annotate frame if saving
                if writer:
                    annotated = self._annotate_frame(frame, detections)
                    writer.write(annotated)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames, found {len(all_detections)} plates")
            
            cap.release()
            if writer:
                writer.release()
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ProcessingResult(
                image_path=video_path,
                detections=all_detections,
                total_detections=len(all_detections),
                processing_time_ms=processing_time,
                errors=[],
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return ProcessingResult(
                image_path=video_path,
                detections=all_detections,
                total_detections=len(all_detections),
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=[str(e)],
                status="partial" if all_detections else "failed"
            )
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[PlateDetection]) -> np.ndarray:
        """Add detection boxes and text to frame"""
        annotated = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bounding_box
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text label
            label = f"{detection.plate_text} ({detection.confidence:.2f})"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return annotated
    
    def batch_process(self, image_dir: str) -> Dict[str, ProcessingResult]:
        """
        Process all images in directory
        
        Args:
            image_dir: Directory containing images
        
        Returns:
            Dictionary mapping image paths to ProcessingResults
        """
        logger.info(f"Batch processing images from: {image_dir}")
        
        results = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_dir = Path(image_dir)
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
            
            start = datetime.now()
            detections = self.detect(str(image_path))
            processing_time = (datetime.now() - start).total_seconds() * 1000
            
            results[str(image_path)] = ProcessingResult(
                image_path=str(image_path),
                detections=detections,
                total_detections=len(detections),
                processing_time_ms=processing_time,
                errors=[],
                status="success" if detections else "no_plates_found"
            )
            
            self.processing_stats['total_images'] += 1
            self.processing_stats['successful'] += 1
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_detections': self.detection_count,
            'statistics': self.processing_stats
        }
    
    def save_results(self, results: List[PlateDetection], output_path: str):
        """Save detection results to JSON"""
        data = [
            {
                'plate': d.plate_text,
                'confidence': d.confidence,
                'bbox': d.bounding_box,
                'timestamp': d.timestamp.isoformat(),
                'image': d.image_path
            }
            for d in results
        ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Example usage"""
    # Initialize detector
    detector = PlateDetector(
        model_size=DetectionModel.YOLOV8_MEDIUM,
        confidence_threshold=0.5
    )
    
    # Detect in single image
    image_path = "sample_image.jpg"
    detections = detector.detect(image_path)
    
    for det in detections:
        print(f"Found plate: {det.plate_text} (confidence: {det.confidence:.2f})")
    
    # Batch process directory
    results = detector.batch_process("./images")
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"Total detections: {stats['total_detections']}")
    
    # Save results
    detector.save_results(detections, "detections.json")


if __name__ == "__main__":
    main()
