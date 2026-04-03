"""
License Plate Detection Module
Handles detection and recognition of license plates using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class LicensePlateDetector:
    """Main detector class for license plates"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.5):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLOv8 model
            confidence: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def detect(self, image_path: str) -> List[dict]:
        """
        Detect license plates in image
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detections with coordinates and confidence
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        results = self.model(image, conf=self.confidence)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detection = {
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist(),
                    'class': result.names[int(box.cls)]
                }
                detections.append(detection)
                
        return detections
    
    def detect_from_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Detect license plates in video
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame, conf=self.confidence)
            annotated_frame = results[0].plot()
            
            if output_path:
                out.write(annotated_frame)
                
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            out.release()
            print(f"Video saved to {output_path}")


if __name__ == "__main__":
    detector = LicensePlateDetector()
    print("Detector initialized successfully")
