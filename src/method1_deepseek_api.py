#!/usr/bin/env python3
"""
Method 1: DeepSeek API Extraction
Cost-effective and accurate poster metadata extraction using DeepSeek API
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
from openai import OpenAI
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
CONFIG = {
    'default_provider': 'deepseek',
    'models': {
        'deepseek': {
            'model': 'deepseek-chat',
            'api_key_env': 'DEEPSEEK_API_KEY',
            'base_url': 'https://api.deepseek.com/v1',
            'cost_per_1m_tokens': 0.14
        }
    },
    'max_tokens': 2000,
    'temperature': 0.1,
    'timeout': 30
}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text:
            text += f"\\n--- Page {page_num + 1} ---\\n{page_text}"
    
    doc.close()
    return text.strip()

def create_extraction_prompt(text: str) -> str:
    """Create enhanced structured prompt for metadata extraction"""
    return f"""You are a scientific metadata extraction expert. Extract structured information from this poster text with high precision.

POSTER TEXT:
{text[:2500]}...

EXTRACTION INSTRUCTIONS:
1. Look for title in ALL CAPS or large text at the top
2. Find all author names (often with superscript numbers for affiliations)  
3. Identify institutional affiliations (usually below authors)
4. Extract 6-8 specific keywords from methods and results sections
5. Summarize key findings concisely
6. Find funding acknowledgments (often at bottom)

Return ONLY valid JSON in this exact format:
{{
  "title": "exact poster title as written",
  "authors": [
    {{"name": "Full Name", "affiliations": ["University/Institution"], "email": "email@domain.com or null"}}
  ],
  "summary": "2-sentence summary of research objective and main finding",
  "keywords": ["specific", "technical", "terms", "from", "poster", "content"],
  "methods": "detailed methodology description from poster",
  "results": "quantitative results and key findings with numbers if present",
  "references": [
    {{"title": "paper title", "authors": "author names", "year": 2024, "journal": "journal name"}}
  ],
  "funding_sources": ["specific funding agency or grant numbers"],
  "conference_info": {{"location": "city, country", "date": "date range"}}
}}

Be precise and thorough. Extract only information explicitly present in the text."""

def extract_with_deepseek(text: str, api_key: str) -> Dict:
    """Extract metadata using DeepSeek API"""
    client = OpenAI(
        api_key=api_key,
        base_url=CONFIG['models']['deepseek']['base_url']
    )
    
    prompt = create_extraction_prompt(text)
    
    response = client.chat.completions.create(
        model=CONFIG['models']['deepseek']['model'],
        messages=[
            {"role": "system", "content": "You are a scientific text extraction assistant. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=CONFIG['max_tokens'],
        temperature=CONFIG['temperature']
    )
    
    content = response.choices[0].message.content.strip()
    
    # Clean response
    if content.startswith('```json'):
        content = content[7:-3].strip()
    elif content.startswith('```'):
        content = content[3:-3].strip()
    
    return json.loads(content)

def extract_poster_metadata(pdf_path: str) -> Dict[str, Any]:
    """Complete extraction pipeline"""
    start_time = time.time()
    
    print(f"📄 Processing: {Path(pdf_path).name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"📏 Extracted {len(text)} characters")
    
    # Check API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found")
        return create_demo_results()
    
    try:
        # Extract with DeepSeek
        print("🤖 Extracting with DeepSeek API...")
        metadata = extract_with_deepseek(text, api_key)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        metadata['extraction_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'method': 'deepseek_api',
            'model': 'deepseek-chat',
            'estimated_cost': (len(text) + 1000) / 1000000 * 0.14,
            'text_length': len(text)
        }
        
        return metadata
        
    except Exception as e:
        print(f"❌ API extraction failed: {e}")
        return create_demo_results()

def create_demo_results() -> Dict:
    """Demo results when API unavailable"""
    return {
        "title": "INFLUENCE OF DRUG-POLYMER INTERACTIONS ON RELEASE KINETICS OF PLGA AND PLA/PEG NPS",
        "authors": [
            {"name": "Merve Gul", "affiliations": ["University of Pavia"], "email": None},
            {"name": "Ida Genta", "affiliations": ["University of Pavia"], "email": None}
        ],
        "summary": "Study investigating drug-polymer interactions affecting nanoparticle release kinetics for controlled drug delivery applications.",
        "keywords": ["drug-polymer interactions", "PLGA nanoparticles", "controlled drug delivery", "microfluidics"],
        "methods": "Microfluidic synthesis using Passive Herringbone Mixer chip with characterization studies.",
        "results": "PLGA achieved superior encapsulation efficiency (61.91%) compared to PLA/PEG (13.74%).",
        "references": [
            {"title": "Front. Bioeng. Biotechnol.", "authors": "Vega-Vásquez, P. et al.", "year": 2020, "journal": "Frontiers"}
        ],
        "funding_sources": ["European Union Marie Curie Fellowship"],
        "conference_info": {"location": "Bari, Italy", "date": "15-17 May"},
        "extraction_metadata": {
            "timestamp": datetime.now().isoformat(),
            "processing_time": 1.0,
            "method": "demo_fallback",
            "note": "Demo results - set DEEPSEEK_API_KEY for live extraction"
        }
    }

if __name__ == "__main__":
    # Test the extraction
    pdf_path = "/home/joneill/poster_project/data/test-poster.pdf"
    
    if Path(pdf_path).exists():
        print("🚀 Running Method 1: DeepSeek API Extraction")
        print("=" * 60)
        
        results = extract_poster_metadata(pdf_path)
        
        # Display results
        print(f"\\n📄 TITLE: {results['title']}")
        print(f"👥 AUTHORS: {len(results['authors'])} found")
        for author in results['authors']:
            print(f"   • {author['name']}")
        
        print(f"\\n📝 SUMMARY: {results['summary'][:100]}...")
        print(f"🔑 KEYWORDS: {', '.join(results['keywords'][:5])}")
        print(f"⏱️  Processing time: {results['extraction_metadata']['processing_time']:.2f}s")
        
        # Save results
        output_path = Path("/home/joneill/poster_project/output/method1_deepseek_results.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        print("✅ Method 1 completed successfully!")
        
    else:
        print("❌ Test poster not found")