#!/usr/bin/env python3
"""
BakLlava Vision Model Troubleshooting Script
============================================

This script tests and troubleshoots the BakLlava vision model for scientific poster extraction.
Based on research: BakLlava combines Mistral 7B with LLaVA 1.5 architecture for multimodal tasks.

Key Points from Research:
- Model ID: "llava-hf/bakLlava-v1-hf" 
- Uses transformers pipeline for "image-text-to-text"
- Requires proper chat template formatting
- Need to handle resource requirements (GPU recommended)
"""

import os
import sys
import json
import fitz  # PyMuPDF
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
from PIL import Image
import io
import re
import requests

# Add environment activation as per user rules
print("🔧 Activating environment...")
os.system("source ~/myenv/bin/activate")

try:
    from transformers import pipeline, LlavaForConditionalGeneration, AutoProcessor
    print("📦 All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("💡 Try: pip install transformers torch pillow pymupdf")
    sys.exit(1)

print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎯 GPU: {torch.cuda.get_device_name()}")

def convert_pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """Convert PDF pages to high-quality PIL Images for vision models"""
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        print(f"📄 Converting PDF to images (DPI: {dpi})...")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Convert to image with specified DPI
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
            print(f"   Page {page_num + 1}: {img.size[0]}x{img.size[1]} pixels")
        
        doc.close()
        return images
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        return []

def create_vision_prompt() -> str:
    """Create the same detailed prompt style as method5 for vision models"""
    return """You are a scientific metadata extraction expert. Analyze this scientific poster image and extract structured information with high precision.

EXTRACTION INSTRUCTIONS:
1. Look for title in ALL CAPS or large text at the top
2. Find all author names (often with superscript numbers for affiliations)
3. Identify institutional affiliations (usually below authors)
4. Extract 6-8 specific keywords from methods and results sections
5. Summarize key findings concisely
6. Find funding acknowledgments (often at bottom) - look for "Acknowledgements" section, grant numbers, Marie Curie fellowships, EU funding
7. Look for references section (usually at bottom in small text) - extract paper titles, authors, years, journals
8. Find conference information - location and dates (often at top or bottom of poster)

IMPORTANT: Look carefully at ALL parts of the poster including small text at the bottom for references and funding information.

Return ONLY valid JSON in this exact format:
{
  "title": "exact poster title as written",
  "authors": [
    {"name": "Full Name", "affiliations": ["University/Institution"], "email": null}
  ],
  "summary": "2-sentence summary of research objective and main finding",
  "keywords": ["specific", "technical", "terms", "from", "poster", "content"],
  "methods": "detailed methodology description from poster",
  "results": "quantitative results and key findings with numbers if present",
  "references": [
    {"title": "paper title", "authors": "author names", "year": 2024, "journal": "journal name"}
  ],
  "funding_sources": ["specific funding agency or grant numbers"],
  "conference_info": {"location": "city, country", "date": "date range"}
}

Be precise and thorough. Extract only information explicitly visible in the poster image."""

def load_bakllava_model():
    """Load BakLlava model using transformers pipeline"""
    print("🤖 Loading BakLlava model...")
    
    model_id = "llava-hf/bakLlava-v1-hf"
    
    try:
        # Try using pipeline with proper dtype
        pipe = pipeline(
            "image-text-to-text", 
            model=model_id, 
            device_map="auto",
            torch_dtype=torch.float16
        )
        print(f"✅ BakLlava loaded successfully via pipeline")
        return pipe, None
        
    except Exception as e:
        print(f"⚠️ Pipeline failed: {e}")
        print("🔄 Trying manual model loading...")
        
        try:
            # Fallback to manual loading
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_id)
            
            print(f"✅ BakLlava loaded manually")
            return model, processor
            
        except Exception as e2:
            print(f"❌ Failed to load BakLlava: {e2}")
            return None, None

def extract_with_bakllava_pipeline(pipe, image: Image.Image, prompt: str) -> str:
    """Extract metadata using BakLlava pipeline (Method 1)"""
    print("🔄 Generating response with BakLlava (Pipeline)...")
    
    try:
        # Prepare conversation format for BakLlava
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Generate response
        outputs = pipe(
            text=messages, 
            max_new_tokens=3000,
            do_sample=False
        )
        
        # Extract text from outputs
        if isinstance(outputs, list) and len(outputs) > 0:
            response = outputs[0].get("generated_text", "")
        elif isinstance(outputs, dict):
            response = outputs.get("generated_text", "")
        else:
            response = str(outputs)
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ Pipeline generation failed: {e}")
        return ""

def extract_with_bakllava_manual(model, processor, image: Image.Image, prompt: str) -> str:
    """Extract metadata using BakLlava manual approach (Method 2)"""
    print("🔄 Generating response with BakLlava (Manual)...")
    
    try:
        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        
        # Apply chat template
        prompt_formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(images=image, text=prompt_formatted, return_tensors="pt").to("cuda")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract only the generated part (remove input)
        response = generated_text.split("ASSISTANT:")[-1].strip()
        
        return response
        
    except Exception as e:
        print(f"❌ Manual generation failed: {e}")
        return ""

def clean_vision_response(response: str) -> Dict[str, Any]:
    """Clean and parse vision response to valid JSON - same as method5"""
    print(f"📝 Raw response length: {len(response)} chars")
    print(f"📝 Raw response preview: {response[:200]}...")
    
    # Remove markdown formatting
    response = response.replace('```json', '').replace('```', '').strip()
    
    # Find JSON object boundaries
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = response[start_idx:end_idx + 1]
        
        # Basic cleanup
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        
        try:
            # Parse and clean the structure
            data = json.loads(json_str)
            
            # Ensure all required fields exist with defaults
            cleaned_data = {
                "title": data.get("title", "Unknown Title"),
                "authors": data.get("authors", []),
                "summary": data.get("summary", "No summary available"),
                "keywords": data.get("keywords", []),
                "methods": data.get("methods", "No methods described"),
                "results": str(data.get("results", "No results available")),
                "references": data.get("references", []),
                "funding_sources": data.get("funding_sources", []),
                "conference_info": data.get("conference_info", {})
            }
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            # Show the problematic JSON for debugging
            print(f"🔍 Problematic JSON: {json_str[:500]}...")
            
            # Return minimal structure if parsing fails
            return {
                "title": "Extraction Failed",
                "authors": [],
                "summary": "Could not parse response",
                "keywords": [],
                "methods": "Parsing error",
                "results": "Parsing error", 
                "references": [],
                "funding_sources": [],
                "conference_info": {},
                "raw_response": response[:1000]  # For debugging
            }
    
    # If no JSON found, return empty structure
    return {
        "title": "No JSON Found",
        "authors": [],
        "summary": "No structured data extracted",
        "keywords": [],
        "methods": "No data",
        "results": "No data",
        "references": [],
        "funding_sources": [],
        "conference_info": {},
        "raw_response": response[:1000]  # For debugging
    }

def test_bakllava_extraction():
    """Main testing function"""
    print("🚀 Testing BakLlava Vision Model")
    print("=" * 50)
    
    # Load model
    model_or_pipe, processor = load_bakllava_model()
    
    if model_or_pipe is None:
        print("❌ Failed to load model - exiting")
        return False
    
    # Convert PDF to image
    pdf_path = "data/test-poster.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    images = convert_pdf_to_images(pdf_path, dpi=300)
    
    if not images:
        print("❌ No images extracted from PDF")
        return False
    
    # Use first page
    image = images[0]
    print(f"📸 Processing image: {image.size[0]}x{image.size[1]} pixels")
    
    # Create prompt
    prompt = create_vision_prompt()
    
    # Test extraction
    start_time = time.time()
    
    # Try pipeline method first
    if hasattr(model_or_pipe, '__call__'):  # It's a pipeline
        response = extract_with_bakllava_pipeline(model_or_pipe, image, prompt)
    else:  # It's a manual model
        response = extract_with_bakllava_manual(model_or_pipe, processor, image, prompt)
    
    end_time = time.time()
    
    if not response:
        print("❌ No response generated")
        return False
    
    # Clean and parse response
    results = clean_vision_response(response)
    
    # Display results
    print("\n📊 EXTRACTION RESULTS:")
    print("=" * 50)
    print(f"📄 TITLE: {results.get('title', 'N/A')}")
    print(f"👥 AUTHORS: {len(results.get('authors', []))} found")
    for i, author in enumerate(results.get('authors', []), 1):
        print(f"   {i}. {author.get('name', 'N/A')} - {author.get('affiliations', ['N/A'])}")
    
    print(f"💰 FUNDING: {results.get('funding_sources', ['None found'])}")
    print(f"📚 REFERENCES: {len(results.get('references', []))} found")
    print(f"⏱️ Processing time: {end_time - start_time:.1f} seconds")
    
    # Save results
    output_path = "output/method5a_bakllava_troubleshoot_results.json"
    os.makedirs("output", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    
    # Check if extraction was successful
    success = (
        results.get('title') not in ['Unknown Title', 'Extraction Failed', 'No JSON Found'] and
        len(results.get('authors', [])) > 0
    )
    
    if success:
        print("✅ BakLlava extraction successful!")
        return True
    else:
        print("⚠️ BakLlava extraction needs improvement")
        if 'raw_response' in results:
            print(f"🔍 Debug - Raw response: {results['raw_response']}")
        return False

if __name__ == "__main__":
    print("🧪 BakLlava Troubleshooting Script")
    print(f"📍 Working directory: {os.getcwd()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - vision models work better with GPU")
        print("🔄 Continuing anyway with CPU...")
    
    success = test_bakllava_extraction()
    
    if success:
        print("\n🎉 Ready to create the 5a notebook!")
    else:
        print("\n🔧 Need to troubleshoot further...")
        print("💡 Check GPU memory, model loading, or prompt formatting")
