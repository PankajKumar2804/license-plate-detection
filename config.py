"""
Configuration management for license plate detection
"""

import json
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration handler"""
    
    # Default settings
    DEFAULTS = {
        "model": {
            "name": "yolov8n",
            "confidence": 0.5,
            "iou_threshold": 0.45
        },
        "ocr": {
            "language": "eng",
            "processor": "tesseract"
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False
        },
        "paths": {
            "models_dir": "./models",
            "output_dir": "./output",
            "data_dir": "./data"
        },
        "performance": {
            "batch_size": 32,
            "max_workers": 4,
            "gpu_enabled": True
        }
    }
    
    def __init__(self, config_file: str = None):
        self.config = self.DEFAULTS.copy()
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, filepath: str):
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                custom_config = json.load(f)
                self._deep_merge(self.config, custom_config)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base"""
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'model.confidence')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary"""
        return self.config.copy()


if __name__ == "__main__":
    config = Config()
    print("Model:", config.get("model.name"))
    print("Confidence:", config.get("model.confidence"))
    config.save_to_file("config.json")
