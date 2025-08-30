#!/usr/bin/env python3
"""
Method 4: Mistral-7B-Instruct Local Extraction
Simple poster metadata extraction using Mistral-7B-Instruct
Uses the same direct prompt style as DeepSeek but runs locally
"""

import os
import json
import fitz  # PyMuPDF
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def normalize_characters(text: str) -> str:
    """Clean up text encoding issues"""
    replacements = {
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2026': '...', '\u00a0': ' ',
        '\u2022': '•', '\u00b0': '°', '\u03b1': 'alpha', '\u03b2': 'beta',
        '\u03bc': 'mu', '\u2264': '<=', '\u2265': '>=', '\u00b1': '±'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and normalize text from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text:
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    
    doc.close()
    return normalize_characters(text.strip())

def create_mistral_prompt(text: str) -> str:
    """Create the same prompt style as DeepSeek for Mistral - using FULL text"""
    return f"""<s>[INST] You are a scientific metadata extraction expert. Extract structured information from this poster text with high precision.

POSTER TEXT:
{text}

EXTRACTION INSTRUCTIONS:
1. Look for title in ALL CAPS or large text at the top
2. Find all author names (often with superscript numbers for affiliations)  
3. Identify institutional affiliations (usually below authors)
4. Extract 6-8 specific keywords from methods and results sections
5. Summarize key findings concisely
6. Find funding acknowledgments (often at bottom) - look for "Acknowledgements" section, grant numbers, Marie Curie fellowships, EU funding

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

Be precise and thorough. Extract only information explicitly present in the text. [/INST]"""

def load_mistral_model():
    """Load Mistral-7B-Instruct model with 8-bit quantization"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print(f"🤖 Loading {model_name}...")
    
    # Configure 8-bit quantization for efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print("✅ Mistral model loaded successfully!")
    return model, tokenizer

def clean_mistral_response(response: str) -> str:
    """Clean Mistral response to extract pure JSON"""
    # Remove common prefixes
    prefixes_to_remove = [
        "Here's the extracted metadata in JSON format:",
        "Here is the extracted metadata:",
        "Based on the poster text, here's the extracted metadata:",
        "The extracted metadata is:",
        "```json",
        "```"
    ]
    
    cleaned = response.strip()
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove trailing ```
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    
    # Find JSON content between braces
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1]
    
    return cleaned

def extract_with_mistral(text: str, model, tokenizer) -> Dict:
    """Extract metadata using Mistral-7B-Instruct"""
    prompt = create_mistral_prompt(text)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("🔄 Generating response...")
    
    # Generate response with optimized parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False,  # Deterministic output
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    generated_text = response[prompt_length:].strip()
    
    print(f"📝 Raw response length: {len(generated_text)} chars")
    
    # Clean and parse JSON
    cleaned_response = clean_mistral_response(generated_text)
    
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"🔍 Cleaned response: {cleaned_response[:500]}...")
        raise

def extract_poster_metadata(pdf_path: str) -> Dict[str, Any]:
    """Complete extraction pipeline using Mistral"""
    start_time = time.time()
    
    print(f"📄 Processing: {Path(pdf_path).name}")
    
    # Extract and normalize text
    text = extract_text_from_pdf(pdf_path)
    print(f"📏 Extracted {len(text)} characters")
    
    # Load model
    model, tokenizer = load_mistral_model()
    
    try:
        # Extract with Mistral
        print("🤖 Extracting with Mistral-7B-Instruct...")
        metadata = extract_with_mistral(text, model, tokenizer)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        metadata['extraction_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'method': 'mistral_7b_instruct_local',
            'model': 'mistralai/Mistral-7B-Instruct-v0.3',
            'text_length': len(text),
            'quantization': '8-bit'
        }
        
        return metadata
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test the extraction
    pdf_path = "data/test-poster.pdf"
    
    if Path(pdf_path).exists():
        print("🚀 Running Method 4: Mistral-7B-Instruct Local Extraction")
        print("=" * 65)
        
        try:
            results = extract_poster_metadata(pdf_path)
            
            # Display results
            print(f"\n📄 TITLE: {results['title']}")
            print(f"👥 AUTHORS: {len(results['authors'])} found")
            for author in results['authors']:
                affiliations = ', '.join(author['affiliations']) if author['affiliations'] else 'None'
                print(f"   • {author['name']} ({affiliations})")
            
            print(f"\n📝 SUMMARY: {results['summary'][:100]}...")
            print(f"🔑 KEYWORDS: {', '.join(results['keywords'][:5])}")
            print(f"💰 FUNDING: {len(results.get('funding_sources', []))} sources")
            print(f"📚 REFERENCES: {len(results.get('references', []))} found")
            print(f"⏱️  Processing time: {results['extraction_metadata']['processing_time']:.2f}s")
            
            # Save results
            output_path = Path("output/method4_mistral_results.json")
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to: {output_path}")
            print("✅ Method 4 completed successfully!")
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            
    else:
        print(f"❌ Test poster not found: {pdf_path}")
