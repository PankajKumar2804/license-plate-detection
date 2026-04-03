"""Basic license plate detection example."""

import asyncio
from detector import PlateDetector

async def main():
    """Run basic detection example."""
    # Initialize detector
    detector = PlateDetector(
        model_path='models/yolov8-plates.pt',
        confidence=0.5,
        gpu=True
    )
    
    # Detect plates in image
    results = detector.detect('sample.jpg')
    
    print(f"Detected {len(results)} plates:")
    for plate in results:
        print(f"  - {plate.text} (confidence: {plate.confidence:.1%})")
        print(f"    Region: {plate.region}, BBox: {plate.bbox}")

if __name__ == "__main__":
    asyncio.run(main())
