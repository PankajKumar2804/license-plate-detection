"""
CLI interface for license plate detection
"""

import argparse
import sys
from pathlib import Path
from src.detector import LicensePlateDetector
from src.recognizer import PlateRecognizer
from config import Config


class LicensePlateCLI:
    """Command-line interface"""
    
    def __init__(self):
        self.config = Config()
        self.detector = LicensePlateDetector(
            confidence=self.config.get("model.confidence", 0.5)
        )
        self.recognizer = PlateRecognizer()
    
    def detect_image(self, image_path: str, output_path: str = None):
        """Detect plates in image"""
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return
        
        print(f"Detecting plates in {image_path}...")
        results = self.detector.detect(image_path)
        
        print(f"Found {len(results)} plates")
        for i, result in enumerate(results):
            print(f"  Plate {i+1}: Confidence {result['confidence']:.2f}")
        
        if output_path:
            print(f"Results saved to {output_path}")
    
    def detect_video(self, video_path: str, output_path: str = None):
        """Detect plates in video"""
        if not Path(video_path).exists():
            print(f"Error: Video not found: {video_path}")
            return
        
        print(f"Processing video: {video_path}")
        self.detector.detect_from_video(video_path, output_path)
        print("Video processing completed")
    
    def batch_detect(self, directory: str):
        """Detect plates in all images in directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Error: Directory not found: {directory}")
            return
        
        image_files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
        print(f"Found {len(image_files)} images")
        
        for image_file in image_files:
            self.detect_image(str(image_file))
    
    def show_config(self):
        """Display current configuration"""
        print("Current Configuration:")
        print("-" * 40)
        config_dict = self.config.to_dict()
        import json
        print(json.dumps(config_dict, indent=2))
    
    def validate_setup(self):
        """Validate detection setup"""
        print("Validating setup...")
        
        try:
            import cv2
            print("✓ OpenCV installed")
        except ImportError:
            print("✗ OpenCV not installed")
        
        try:
            from ultralytics import YOLO
            print("✓ YOLOv8 installed")
        except ImportError:
            print("✗ YOLOv8 not installed")
        
        try:
            import pytesseract
            print("✓ Tesseract installed")
        except ImportError:
            print("✗ Tesseract not installed")


def main():
    parser = argparse.ArgumentParser(
        description='License Plate Detection CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python cli.py detect image.jpg
  python cli.py detect video.mp4 -o output.mp4
  python cli.py batch /path/to/images
  python cli.py config
  python cli.py validate
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect plates in image/video')
    detect_parser.add_argument('input', help='Input image or video file')
    detect_parser.add_argument('-o', '--output', help='Output file path')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process directory')
    batch_parser.add_argument('directory', help='Directory containing images')
    
    # Config command
    subparsers.add_parser('config', help='Show configuration')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate setup')
    
    args = parser.parse_args()
    
    cli = LicensePlateCLI()
    
    if args.command == 'detect':
        if args.input.endswith(('.mp4', '.avi', '.mov')):
            cli.detect_video(args.input, args.output)
        else:
            cli.detect_image(args.input, args.output)
    
    elif args.command == 'batch':
        cli.batch_detect(args.directory)
    
    elif args.command == 'config':
        cli.show_config()
    
    elif args.command == 'validate':
        cli.validate_setup()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
