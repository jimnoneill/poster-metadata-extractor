#!/usr/bin/env python3
"""
Method 2: Qwen Local Extraction
Local small language model for cost-effective poster metadata extraction
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import fitz  # PyMuPDF
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
import os
import logging

# Suppress all warnings and errors
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Suppress protobuf and transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Try to suppress specific protobuf MessageFactory warnings
try:
    import google.protobuf.message
    # Monkey patch to suppress the GetPrototype AttributeError
    if hasattr(google.protobuf.message, 'MessageFactory'):
        original_init = google.protobuf.message.MessageFactory.__init__
        def patched_init(self):
            try:
                original_init(self)
            except AttributeError:
                pass
        google.protobuf.message.MessageFactory.__init__ = patched_init
except (ImportError, AttributeError):
    pass

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text:
            text += f"\\n--- Page {page_num + 1} ---\\n{page_text}"
    
    doc.close()
    return text.strip()

class QwenExtractor:
    """Qwen2.5-1.5B-Instruct based metadata extractor"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"📥 Loading {model_name}...")
        
        # Load tokenizer with stderr redirection
        import sys
        from contextlib import redirect_stderr
        import io
        
        with redirect_stderr(io.StringIO()):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if CUDA available
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                # CPU loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                device = torch.device('cpu')
                self.model = self.model.to(device)
        
        self.model.eval()
        print(f"✅ Model loaded successfully")
    
    def extract_field(self, text: str, field: str) -> Any:
        """Extract specific field using few-shot prompting"""
        
        prompts = {
            'title': f"""Extract the title from this poster text:

Text: "{text[:500]}"

Title:""",
            
            'authors': f"""Extract author names (comma-separated) from this poster:

Text: "{text[:500]}"

Authors:""",
            
            'summary': f"""Write a 2-sentence summary of this poster:

Text: "{text[:800]}"

Summary:""",
            
            'keywords': f"""Extract 5-6 keywords from this poster:

Text: "{text[:600]}"

Keywords:""",
            
            'methods': f"""Extract the main methods from this research:

Text: "{text[:800]}"

Methods:""",
            
            'results': f"""Extract the main results from this poster:

Text: "{text[:800]}"

Results:"""
        }
        
        if field not in prompts:
            return ""
        
        prompt = prompts[field]
        
        # Create chat template
        messages = [
            {"role": "system", "content": "Extract information precisely as requested."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse response based on field
        if field == 'authors':
            authors = [a.strip() for a in response.split(',') if a.strip()]
            return [{'name': author} for author in authors[:6]]  # Limit to 6
        elif field == 'keywords':
            keywords = [k.strip() for k in response.split(',') if k.strip()]
            return keywords[:8]  # Limit to 8
        else:
            return response.strip()

def extract_poster_metadata_qwen(pdf_path: str) -> Dict[str, Any]:
    """Extract metadata using Qwen model"""
    start_time = time.time()
    
    print(f"📄 Processing: {Path(pdf_path).name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"📏 Extracted {len(text)} characters")
    
    try:
        # Initialize extractor
        print("🤖 Initializing Qwen2.5-1.5B model...")
        extractor = QwenExtractor()
        
        # Extract each field
        print("🔍 Extracting metadata components...")
        
        metadata = {
            'title': extractor.extract_field(text, 'title'),
            'authors': extractor.extract_field(text, 'authors'),
            'summary': extractor.extract_field(text, 'summary'),
            'keywords': extractor.extract_field(text, 'keywords'),
            'methods': extractor.extract_field(text, 'methods'),
            'results': extractor.extract_field(text, 'results'),
            'references': [],  # Would need more complex extraction
            'funding_sources': [],  # Would need pattern matching
            'conference_info': {'location': None, 'date': None},
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'method': 'qwen_local',
                'model': 'Qwen2.5-1.5B-Instruct',
                'device': str(next(extractor.model.parameters()).device),
                'text_length': len(text)
            }
        }
        
        return metadata
        
    except Exception as e:
        print(f"❌ Qwen extraction failed: {e}")
        return {
            'error': str(e),
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'method': 'qwen_local_failed',
                'processing_time': time.time() - start_time
            }
        }

if __name__ == "__main__":
    # Test the extraction
    pdf_path = "../data/test-poster.pdf"
    
    if Path(pdf_path).exists():
        print("🚀 Running Method 2: Qwen Local Extraction")
        print("=" * 60)
        
        results = extract_poster_metadata_qwen(pdf_path)
        
        if 'error' not in results:
            # Display results
            print(f"\\n📄 TITLE: {results['title']}")
            print(f"👥 AUTHORS: {len(results['authors'])} found")
            for author in results['authors']:
                print(f"   • {author['name']}")
            
            print(f"\\n📝 SUMMARY: {results['summary'][:100]}...")
            print(f"🔑 KEYWORDS: {', '.join(results['keywords'][:5])}")
            print(f"⏱️  Processing time: {results['extraction_metadata']['processing_time']:.2f}s")
            
            # Save results
            output_path = Path("../output/method2_qwen_results.json")
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to: {output_path}")
            print("✅ Method 2 completed successfully!")
        else:
            print(f"❌ Extraction failed: {results['error']}")
        
    else:
        print("❌ Test poster not found")

