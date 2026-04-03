"""Setup configuration for license-plate-detection."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="license-plate-detection",
    version="1.0.0",
    author="Pankaj Kumar",
    description="Real-time license plate detection and recognition system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PankajKumar2804/license-plate-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
    ],
)
