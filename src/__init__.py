"""
Scientific Poster Metadata Extraction Pipeline

A comprehensive system for extracting structured metadata from scientific posters 
using Large Language Models and document processing techniques.
"""

__version__ = "1.0.0"
__author__ = "Poster Extraction Pipeline"
__email__ = "contact@example.com"

from .extract_metadata import extract_poster_metadata
from .pdf_processor import PDFProcessor  
from .llm_extractor import LLMExtractor
from .validator import MetadataValidator

__all__ = [
    "extract_poster_metadata",
    "PDFProcessor", 
    "LLMExtractor",
    "MetadataValidator"
]

