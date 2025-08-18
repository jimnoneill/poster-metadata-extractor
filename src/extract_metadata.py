#!/usr/bin/env python3
"""
Main extraction script for poster metadata extraction pipeline.
This script can be run from command line or imported as a module.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.pdf_processor import PDFProcessor
from src.llm_extractor import LLMExtractor  
from src.validator import MetadataValidator


def extract_poster_metadata(pdf_path: str, output_path: Optional[str] = None, 
                          llm_provider: str = "openai") -> Dict:
    """
    Extract structured metadata from a scientific poster PDF.
    
    Args:
        pdf_path: Path to the input PDF file
        output_path: Optional path to save the extracted JSON
        llm_provider: LLM provider to use ("openai" or "anthropic")
        
    Returns:
        Dictionary containing extracted metadata
    """
    
    # Initialize components
    pdf_processor = PDFProcessor()
    llm_extractor = LLMExtractor(provider=llm_provider)
    validator = MetadataValidator()
    
    try:
        # Step 1: Extract text from PDF
        print(f"🚀 Starting metadata extraction for: {pdf_path}")
        text, pdf_metadata = pdf_processor.process_pdf(pdf_path)
        
        # Step 2: Extract metadata using LLM
        raw_metadata = llm_extractor.extract_metadata(text)
        
        # Step 3: Validate extraction
        is_valid, confidence_scores, errors = validator.validate_and_score(raw_metadata, text)
        
        # Add extraction metadata
        raw_metadata['extraction_metadata'] = {
            "processing_time": 0.0,  # Will be set by caller
            "model_version": llm_extractor.model,
            "validation_passed": is_valid,
            "confidence_scores": confidence_scores,
            "validation_errors": errors
        }
        
        # Save output if requested
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(raw_metadata, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {output_path}")
        
        return raw_metadata
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise


def main():
    """Command line interface for the extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract structured metadata from scientific poster PDFs"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to output JSON file (optional)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Run extraction
        results = extract_poster_metadata(
            pdf_path=args.input,
            output_path=args.output,
            llm_provider=args.provider
        )
        
        if args.verbose:
            print("\n" + "="*60)
            print("EXTRACTION SUMMARY")
            print("="*60)
            print(f"Title: {results.get('title', 'N/A')}")
            print(f"Authors: {len(results.get('authors', []))}")
            print(f"Keywords: {len(results.get('keywords', []))}")
            print(f"References: {len(results.get('references', []))}")
            
        print("\n✅ Extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


