#!/usr/bin/env python3
"""
Setup script for the Poster Metadata Extraction Pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "PyMuPDF>=1.23.0",
        "pdfplumber>=0.9.0", 
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "jsonschema>=4.19.0",
        "pydantic>=2.5.0"
    ]

setup(
    name="poster-metadata-extractor",
    version="1.0.0",
    author="Technical Assessment",
    author_email="assessment@example.com",
    description="Extract structured metadata from scientific posters using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/poster-metadata-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "extract-poster=src.extract_metadata:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)


