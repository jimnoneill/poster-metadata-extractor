#!/usr/bin/env python3
"""
Method 5: Qwen2-VL Vision-Language Model Direct Extraction
Direct image processing for scientific posters using genuine vision-based extraction
Uses the same direct prompt style as Mistral but for vision input
"""

import os
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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def convert_pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """Convert PDF pages to high-quality PIL Images for vision models"""
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

def create_direct_vision_prompt() -> str:
    """Create a direct prompt that produces clean JSON output"""
    return """Analyze this scientific poster image and extract metadata. Return ONLY valid JSON with no explanations or formatting:

{
  "title": "exact poster title",
  "authors": [
    {"name": "Full Name", "affiliations": ["Institution"], "email": null}
  ],
  "summary": "brief research summary",
  "keywords": ["key", "terms"],
  "methods": "methodology description",
  "results": "main findings and results",
  "references": [
    {"title": "paper title", "authors": "authors", "year": 2024, "journal": "journal"}
  ],
  "funding_sources": ["funding info"],
  "conference_info": {"location": "location", "date": "date"}
}"""

def load_qwen2vl_model():
    """Load Qwen2-VL model and processor"""
    print("🤖 Loading Qwen2-VL model...")
    
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"✅ Qwen2-VL loaded successfully")
        return model, processor
        
    except Exception as e:
        print(f"❌ Failed to load Qwen2-VL: {e}")
        return None, None

def extract_with_qwen2vl(model, processor, image: Image.Image, prompt: str) -> str:
    """Extract metadata using Qwen2-VL vision model"""
    print("🔄 Generating response with Qwen2-VL...")
    
    try:
        # Prepare conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to("cuda")
        
        # Generate response with strict parameters for JSON
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1500,  # Reduced for cleaner output
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Trim input tokens and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return ""

def clean_vision_response(response: str) -> Dict[str, Any]:
    """Clean and parse vision response to valid JSON"""
    print(f"📝 Raw response length: {len(response)} chars")
    
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
                "results": str(data.get("results", "No results available"))[:500],  # Ensure string
                "references": data.get("references", [])[:3],  # Limit to 3 references
                "funding_sources": data.get("funding_sources", []),
                "conference_info": data.get("conference_info", {})
            }
            
            # Clean references to expected format
            if cleaned_data["references"]:
                fixed_refs = []
                for ref in cleaned_data["references"]:
                    if isinstance(ref, dict):
                        fixed_ref = {
                            "title": str(ref.get('title', 'Unknown title')),
                            "authors": str(ref.get('authors', 'Unknown authors')),
                            "year": int(ref.get('year', 2024)) if isinstance(ref.get('year'), (int, str)) else 2024,
                            "journal": str(ref.get('journal', 'Unknown journal'))
                        }
                        fixed_refs.append(fixed_ref)
                cleaned_data["references"] = fixed_refs
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
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
                "conference_info": {}
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
        "conference_info": {}
    }

def main():
    print("🚀 Running Method 5: Qwen2-VL Vision Direct Extraction")
    print("=" * 65)
    
    # Check CUDA
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
    
    # Load model
    model, processor = load_qwen2vl_model()
    if model is None:
        print("❌ Failed to load model, exiting...")
        return
    
    # Convert PDF to image
    pdf_path = "data/test-poster.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    images = convert_pdf_to_images(pdf_path, dpi=200)
    if not images:
        print("❌ No images extracted from PDF")
        return
    
    # Use first page
    image = images[0]
    print(f"📸 Processing image: {image.size[0]}x{image.size[1]} pixels")
    
    # Create prompt
    prompt = create_direct_vision_prompt()
    
    # Extract metadata
    start_time = time.time()
    response = extract_with_qwen2vl(model, processor, image, prompt)
    end_time = time.time()
    
    if not response:
        print("❌ No response generated")
        return
    
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
    output_path = "output/method5_qwen2vl_vision_results.json"
    os.makedirs("output", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    print("✅ Method 5 completed successfully!")
    print("🎯 Vision approach - processes images without text extraction!")

if __name__ == "__main__":
    main()
