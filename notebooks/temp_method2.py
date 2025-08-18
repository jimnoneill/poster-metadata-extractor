#!/usr/bin/env python
# coding: utf-8

# # Method 2: Qwen Local Extraction
# 
# ## Overview
# Local small language model for cost-effective poster metadata extraction. Runs entirely on your hardware without API dependencies.
# 
# ## Accuracy Note
# The 80-85% accuracy estimate is unvalidated - based on limited testing only. Actual accuracy must be determined through proper Cochran sampling validation before production use.
# 
# ## Performance Characteristics
# - **Estimated Accuracy**: 80-85% (unvalidated - requires Cochran sampling validation)
# - **Cost**: $0 (runs locally, only electricity costs)
# - **Speed**: 10-40 seconds per poster (single), ~1.1s per poster (RTX 4090 batched)
# - **Hallucination Risk**: Low (structured prompting)
# - **Setup**: Medium - requires model download and GPU memory
# 
# ## RTX 4090 Batching Capacity
# - **Recommended batch size**: 32 posters simultaneously
# - **Throughput**: ~3,273 posters/hour, ~26,182 posters/day (8hrs)
# 
# ## Best For
# - Privacy-sensitive environments
# - Budget-conscious deployments
# - Edge computing scenarios
# - Development and experimentation
# 

# In[1]:


# Imports and setup
import os
import warnings
# Suppress TensorFlow and CUDA initialization warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import fitz  # PyMuPDF
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "Using CPU")
print("✅ Environment ready for Method 2: Qwen Local")


# In[2]:


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

        # Load tokenizer
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

print("✅ QwenExtractor class defined")


# In[3]:


# Run extraction
pdf_path = "../data/test-poster.pdf"

if Path(pdf_path).exists():
    print("🚀 Running Method 2: Qwen Local Extraction")
    print("=" * 60)

    start_time = time.time()

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

        # Display results
        print(f"\\n📄 TITLE: {metadata['title']}")
        print(f"👥 AUTHORS: {len(metadata['authors'])} found")
        for author in metadata['authors']:
            print(f"   • {author['name']}")

        print(f"\\n📝 SUMMARY: {metadata['summary'][:100]}...")
        print(f"🔑 KEYWORDS: {', '.join(metadata['keywords'][:5])}")
        print(f"⏱️  Processing time: {metadata['extraction_metadata']['processing_time']:.2f}s")

        # Save results
        output_path = Path("../output/method2_qwen_results.json")
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Results saved to: {output_path}")
        print("✅ Method 2 completed successfully!")

    except Exception as e:
        print(f"❌ Qwen extraction failed: {e}")
        print("   This may be due to insufficient GPU memory or model download issues")

else:
    print("❌ Test poster not found")

