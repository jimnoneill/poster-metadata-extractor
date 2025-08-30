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

# In[1]:


# Imports and setup
import os
import warnings
import contextlib
import io
import logging
# Suppress TensorFlow and CUDA initialization warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
#from jtools import normalize_characters
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import fitz  # PyMuPDF
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")
print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "Using CPU")
print("✅ Environment ready for Method 2: Qwen Local")


# In[2]:


import unicodedata
import re
def remove_quotes(text):
    """Remove surrounding quotes from text"""
    text = text.strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    return text

def clean_qwen_response(response: str, field_type: str) -> str:
    """Clean up verbose Qwen responses to extract just the content"""
    response = response.strip()

    # Remove common verbose prefixes
    prefixes_to_remove = [
        "The title of the poster is:",
        "Here are the author names extracted in a comma-separated list:",
        "Here is a 2-sentence summary of the poster:",
        "Here are 5-6 keywords extracted from the poster:",
        "Here are the methods mentioned in the poster:",
        "Here are the main results extracted from the poster:",
        "Here are the references found in the poster:",
        "Here are the funding sources found:",
        "Here is the conference information:",
        "The title is:",
        "Authors:",
        "Summary:",
        "Keywords:",
        "Methods:",
        "Results:",
        "References:",
        "Funding:",
        "Conference:"
    ]

    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()

    # Remove numbered lists (1., 2., etc.)
    if field_type in ['keywords', 'methods', 'funding_sources']:
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove numbering like "1.", "2.", "*", "-" at start of line
            line = re.sub(r'^\s*[\d]+\.\s*', '', line)
            line = re.sub(r'^\s*[\*\-]\s*', '', line)
            if line.strip():
                cleaned_lines.append(line.strip())
        response = '\n'.join(cleaned_lines) if field_type == 'methods' else ', '.join(cleaned_lines)

    # Remove quotes and extra whitespace
    response = remove_quotes(response)

    return response.strip()
def normalize_characters(text):
    # Normalize Greek characters
    greek_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']
    for char in greek_chars:
        text = text.replace(char, unicodedata.normalize('NFC', char))

    # Normalize space characters
    space_chars = ['\xa0', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u202f', '\u205f', '\u3000']
    for space in space_chars:
        text = text.replace(space, ' ')

    # Normalize single quotes
    single_quotes = ['‘', '’', '‛', '′', '‹', '›', '‚', '‟']
    for quote in single_quotes:
        text = text.replace(quote, "'")

    # Normalize double quotes
    double_quotes = ['“', '”', '„', '‟', '«', '»', '〝', '〞', '〟', '＂']
    for quote in double_quotes:
        text = text.replace(quote, '"')

    # Normalize brackets
    brackets = {
        '【': '[', '】': ']',
        '（': '(', '）': ')',
        '｛': '{', '｝': '}',
        '〚': '[', '〛': ']',
        '〈': '<', '〉': '>',
        '《': '<', '》': '>',
        '「': '[', '」': ']',
        '『': '[', '『': ']',
        '〔': '[', '〕': ']',
        '〖': '[', '〗': ']'
    }
    for old, new in brackets.items():
        text = text.replace(old, new)

    # Normalize hyphens and dashes
    hyphens_and_dashes = ['‐', '‑', '‒', '–', '—', '―']
    for dash in hyphens_and_dashes:
        text = text.replace(dash, '-')

    # Normalize line breaks
    line_breaks = ['\r\n', '\r']
    for line_break in line_breaks:
        text = text.replace(line_break, '\n')

    # Normalize superscripts and subscripts to normal numbers
    superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    subscripts = '₀₁₂₃₄₅₆₇₈₉'
    normal_numbers = '0123456789'

    for super_, sub_, normal in zip(superscripts, subscripts, normal_numbers):
        text = text.replace(super_, normal).replace(sub_, normal)

    # Remove or normalize any remaining special characters using the 'NFKD' method
    text = unicodedata.normalize('NFKD', text)

    return remove_quotes(text)


# In[3]:


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    doc = fitz.open(pdf_path)
    text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text:
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"

    doc.close()

    # Apply normalize_characters to the ENTIRE extracted text
    text = normalize_characters(text)
    return text.strip()

class QwenExtractor:
    """Qwen2.5-1.5B-Instruct based metadata extractor"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"📥 Loading {model_name}...")

        # Load tokenizer with stderr suppression
        with contextlib.redirect_stderr(io.StringIO()):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization if CUDA available
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )

            # Load model with stderr suppression
            with contextlib.redirect_stderr(io.StringIO()):
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
            device = torch.device("cpu")
            self.model = self.model.to(device)

        self.model.eval()
        print(f"✅ Model loaded successfully")

    def extract_references_two_step(self, text: str) -> List[Dict[str, Any]]:
        """Two-step references extraction: find section, then parse individual refs"""

        # Step 1: Find the references section
        find_refs_prompt = f"""Find the references section in this poster text. Look for sections titled "References", "Bibliography", "Citations", or numbered lists at the bottom.

Extract ONLY the references section text. Include all numbered entries like [1], [2] or 1., 2. etc.

If no clear references section exists, return "No references section found".

Text: "{text[-2500:]}"

References Section Text:"""

        messages = [
            {"role": "system", "content": "You are a precise text extraction assistant. Extract only the requested section without explanatory text."},
            {"role": "user", "content": find_refs_prompt}
        ]

        # Get references section
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)

        refs_section = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if "no references section found" in refs_section.lower() or len(refs_section) < 10:
            return []

        # Step 2: Parse individual references from the section
        parse_refs_prompt = f"""Parse these references into structured format. Each reference should be extracted as:
"Title - Authors (Year) Journal"

Examples of what you might see and how to parse:
- Input: "1. Smith, J. et al. Drug delivery systems. Nature 2023; 45(2):123-134."
- Output: "Drug delivery systems - Smith, J. et al. (2023) Nature"

- Input: "[2] Jones M, Wilson K. PLGA nanoparticles. Advanced Materials 2022, 34(5)."  
- Output: "PLGA nanoparticles - Jones M, Wilson K (2022) Advanced Materials"

Parse each numbered reference and separate with " | ".

References text to parse:
"{refs_section}"

Parsed References:"""

        messages = [
            {"role": "system", "content": "You are a citation parsing specialist. Extract structured references in the exact format requested."},
            {"role": "user", "content": parse_refs_prompt}
        ]

        # Parse references
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)

        parsed_refs = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # CRITICAL: Enhanced response cleaning for two-step method
        parsed_refs = clean_qwen_response(parsed_refs, "references")

        # Additional cleaning for verbose prefixes
        if "here are the parsed references in the desired format:" in parsed_refs.lower():
            lines = parsed_refs.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if (line and 
                    not line.lower().startswith('here are the parsed references') and
                    not line.lower().startswith('here are') and
                    line != ''):
                    cleaned_lines.append(line)
            parsed_refs = ' '.join(cleaned_lines)

        parsed_refs = parsed_refs.strip('"').strip("'").strip()

        # CRITICAL: Apply response cleaning to remove verbose prefixes
        parsed_refs = clean_qwen_response(parsed_refs, "references")

        # Debug logging
        print(f"🔍 DEBUG - Raw parsed_refs: {parsed_refs[:100]}...")

        # Process the parsed references
        references = []
        if "|" in parsed_refs:
            ref_parts = parsed_refs.split("|")
        else:
            ref_parts = [parsed_refs] if parsed_refs else []

        for ref_part in ref_parts:
            ref_part = ref_part.strip()
            if not ref_part or len(ref_part) < 5:
                continue

            title = ""
            authors = ""
            year = None
            journal = ""

            # Parse "Title - Authors (Year) Journal" format
            if " - " in ref_part:
                title_part, rest = ref_part.split(" - ", 1)
                title = title_part.strip()

                # Look for (Year) pattern
                import re
                year_match = re.search(r'\((\d{4})\)', rest)
                if year_match:
                    year = int(year_match.group(1))
                    before_year = rest[:year_match.start()].strip()
                    after_year = rest[year_match.end():].strip()
                    authors = before_year
                    journal = after_year
                else:
                    authors = rest
            else:
                title = ref_part

            references.append({
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal
            })

            if len(references) >= 5:  # Limit to 5
                break

        return references


    def enhanced_author_parsing(self, response: str) -> list:
        """Enhanced author parsing to handle complex nested parentheses"""
        import re

        def split_respecting_parentheses(text, separator=" | "):
            """Split by separator but ignore separators inside parentheses"""
            parts = []
            current_part = ""
            paren_depth = 0
            i = 0

            while i < len(text):
                char = text[i]

                if char == '(':
                    paren_depth += 1
                    current_part += char
                elif char == ')':
                    paren_depth -= 1
                    current_part += char
                elif text[i:i+len(separator)] == separator and paren_depth == 0:
                    if current_part.strip():
                        parts.append(current_part.strip())
                    current_part = ""
                    i += len(separator) - 1
                else:
                    current_part += char

                i += 1

            if current_part.strip():
                parts.append(current_part.strip())

            return parts

        # Split the response into author entries
        author_entries = split_respecting_parentheses(response)

        authors = []

        for entry in author_entries:
            entry = entry.strip()
            if not entry:
                continue

            name = ""
            affiliations = []

            if '(' in entry and ')' in entry:
                paren_start = entry.find('(')
                name = entry[:paren_start].strip()

                affiliation_start = paren_start + 1
                paren_count = 1
                affiliation_end = -1
                for j in range(affiliation_start, len(entry)):
                    if entry[j] == '(':
                        paren_count += 1
                    elif entry[j] == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            affiliation_end = j
                            break

                if affiliation_end > affiliation_start:
                    affiliation_text = entry[affiliation_start:affiliation_end]
                    affiliation_parts = affiliation_text.split(' | ')
                    for part in affiliation_parts:
                        part = part.strip()
                        if part and len(part) > 2:
                            affiliations.append(part)
            else:
                name = entry

            if not name:
                continue

            # Enhanced filtering
            institutional_keywords = [
                'department', 'university', 'institute', 'center', 'centre', 
                'school', 'college', 'laboratory', 'lab', 'division',
                'research', 'faculty', 'hospital', 'clinic', 'universitat',
                'catalunya', 'politecnica', 'upc'
            ]

            name_lower = name.lower()

            if any(keyword in name_lower for keyword in institutional_keywords):
                continue

            if name_lower.startswith(('of ', 'for ', 'and ', 'the ', 'de ', 'del ')):
                continue

            name_words = name.split()
            if len(name_words) == 1:
                if not (name.istitle() and 3 <= len(name) <= 15):
                    continue

            if len(name_words) >= 2 or (len(name_words) == 1 and name.istitle() and 3 <= len(name) <= 15):
                authors.append({
                    "name": name,
                    "affiliations": affiliations,
                    "email": None
                })

                if len(authors) >= 6:
                    break

        return authors

    def extract_field(self, text: str, field: str) -> Any:
        """Extract specific field using few-shot prompting"""

        # More explicit prompts that discourage verbose responses
        prompts = {
            'title': f"""Extract only the title from this poster text. Provide just the title text, nothing else.

Text: "{text[:500]}"

Title:""",

            'authors': f"""Extract the complete author names and their affiliations from this poster. Look for the author section near the title, and match superscript numbers to institutions listed nearby.

IMPORTANT: Extract COMPLETE names (first and last name) and link them to their affiliations using superscript numbers.

Format as: "Author Name (Affiliation)" for each author, separated by " | "

Examples:
- If you see "Merve Gul¹, Ida Genta²" and "¹University of Pavia, ²University of Rome", extract:
  "Merve Gul (University of Pavia) | Ida Genta (University of Rome)"
- If you see "John Smith¹'², Mary Johnson³" and affiliations listed, match the numbers.

Text: "{text[:1200]}"

Authors with Affiliations:""",

            'summary': f"""Write a concise 2-sentence summary of this poster's research. Be direct and factual.

Text: "{text[:800]}"

Summary:""",

            'keywords': f"""Extract 5-6 key technical terms from this poster. List only the keywords separated by commas.

Text: "{text[:600]}"

Keywords:""",

            'methods': f"""Extract the research methods described in this poster. Be concise and specific.

Text: "{text[:800]}"

Methods:""",

            'results': f"""Extract the main research findings from this poster. Include specific numbers/measurements if present.

Text: "{text[:800]}"

Results:""",

            'references': f"""Extract references or citations from this poster. Look for sections titled "References", "Bibliography", "Citations", or numbered reference lists, usually at the bottom. Look for patterns like:
- [1], [2], [3] followed by citation details
- 1., 2., 3. followed by publication info
- Author names followed by titles and years
- Journal names, publication years, volume/page numbers

Extract complete references in the format: "Title - Authors (Year) Journal" separated by " | " for multiple references.

Examples:
- "Drug delivery systems - Smith et al. (2023) Nature"
- "PLGA nanoparticles - Jones, M. et al. (2022) Advanced Materials"

If you find numbered references but can't parse full details, extract what's available.

Text: "{text[:2000]}"

References:""",

            'funding_sources': f"""Extract funding information from this poster text. Look specifically for ACKNOWLEDGMENTS/ACKNOWLEDGEMENTS sections (usually at bottom of poster).

Common funding patterns to find:
• "We acknowledge...", "The authors acknowledge...", "This work was supported by..."
• "Financial support from...", "Funded by...", "Grant support from..."
• Grant numbers: "Grant No. XXXXX", "Project #XXXXX", "#XXXXX"
• Funding agencies: "NSF", "NIH", "EU", "Horizon 2020", "ERC", "EPSRC", etc.
• University/institutional funding, Fellowship acknowledgments

Look for sections with words like: "acknowledge", "support", "funding", "grant", "fellowship", "financial", "sponsored", "contract", "award"

Extract specific funding sources, grant numbers, or agencies. List them separated by commas.
If no funding information found, return "None found".

Text: "{text}"

Funding/Acknowledgments:""",

            'conference_info': f"""Extract conference information from this poster. Look for location names (cities, countries) and dates. This information is often at the bottom of the poster or near the title.

Look for patterns like:
- City names (Bari, Rome, Paris, etc.)
- Countries (Italy, France, USA, etc.) 
- Dates (May 15-17, June 2024, etc.)

Format as: "Location: City, Country | Date: date range" or just the location/date if found.

Text: "{text[:1200]}"

Conference Info:"""
        }

        if field not in prompts:
            return ""

        prompt = prompts[field]

        # Create chat template with PDF context and explicit instructions for conciseness
        messages = [
            {"role": "system", "content": "You are a precise data extraction assistant working with unstructured text converted from conference poster PDFs. The text may have formatting issues, scattered layout, and mixed content. Focus on extracting the specific requested information. Provide only the requested information without explanatory text, prefixes, or formatting. Be direct and concise."},
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

        # Generate with greedy decoding for deterministic output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,     # Greedy decoding = deterministic (most probable token)
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Prevent repetition
                # Note: temperature/top_p not needed with do_sample=False
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Clean up the response
        response = clean_qwen_response(response, field)

        # Debug logging for authors
        if field == "authors":
            print(f"🔍 DEBUG - Raw author response: {response[:200]}...")

        # Parse response based on field
        if field == "authors":
            authors = []
            # Handle format: "Author Name (Affiliation) | Author Name (Affiliation)"
            if "|" in response:
                author_entries = response.split("|")
            else:
                # Fallback: try comma separation
                author_entries = response.split(",")

            for entry in author_entries:
                entry = entry.strip()
                if not entry:
                    continue

                name = ""
                affiliations = []

                # Try to parse "Name (Affiliation)" format
                if "(" in entry and ")" in entry:
                    name_part = entry.split("(")[0].strip()
                    affiliation_part = entry.split("(")[1].split(")")[0].strip()

                    # Clean up the name
                    name = name_part

                    # Add affiliation if it looks valid
                    if affiliation_part and len(affiliation_part) > 2:
                        affiliations.append(affiliation_part)
                else:
                    # No parentheses, just the name
                    name = entry

                # Filter out non-author entries
                institutional_keywords = [
                    'department', 'university', 'institute', 'center', 'centre', 
                    'school', 'college', 'laboratory', 'lab', 'division',
                    'research', 'faculty', 'hospital', 'clinic'
                ]

                name_lower = name.lower()

                # Skip if the "name" is clearly an institution
                if any(keyword in name_lower for keyword in institutional_keywords):
                    continue

                # Skip if it starts with prepositions or looks like an affiliation
                if (name_lower.startswith(('of ', 'for ', 'and ', 'the ')) or
                    len(name.split()) > 4):  # Names shouldn't be too long
                    continue

                if name and len(name.split()) >= 2:  # Should have at least first and last name
                    authors.append({
                        "name": name,
                        "affiliations": affiliations,
                        "email": None
                    })

                    if len(authors) >= 6:  # Limit to 6 authors
                        break

            return authors

        elif field == "keywords":
            keywords = [k.strip() for k in response.split(",") if k.strip()]
            return keywords[:8]  # Limit to 8

        elif field == "references":
            # Use the new two-step method for better references extraction
            return self.extract_references_two_step(text)

        elif field == "funding_sources":
            if response.lower() == "none found" or not response.strip():
                return []

            funding = [f.strip() for f in response.split(",") if f.strip() and f.strip().lower() != "none found"]
            return funding[:5]  # Limit to 5

        elif field == "conference_info":
            if response.lower() == "none found" or not response.strip():
                return {"location": None, "date": None}

            location = None
            date = None

            # Handle format: "Location: City, Country | Date: date range"
            if "|" in response:
                parts = response.split("|")
                for part in parts:
                    part = part.strip()
                    if part.lower().startswith("location:"):
                        location = part.split(":", 1)[1].strip()
                    elif part.lower().startswith("date:"):
                        date = part.split(":", 1)[1].strip()
            else:
                # Try to detect location/date in single string
                if any(word in response.lower() for word in ["location", "city", "country"]):
                    location = response.strip()
                elif any(word in response.lower() for word in ["date", "may", "june", "july", "august", "september"]):
                    date = response.strip()
                else:
                    # Assume it's location if no clear indicator
                    location = response.strip()

            return {"location": location, "date": date}
        else:
            return response

print("✅ QwenExtractor class defined")


# In[4]:


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

        title = extractor.extract_field(text, "title")
        authors = extractor.extract_field(text, "authors")
        summary = extractor.extract_field(text, "summary")
        keywords = extractor.extract_field(text, "keywords")
        methods = extractor.extract_field(text, "methods")
        results_text = extractor.extract_field(text, "results")
        references = extractor.extract_field(text, "references")
        funding_sources = extractor.extract_field(text, "funding_sources")
        conference_info = extractor.extract_field(text, "conference_info")

        # Compile results
        results = {
            "title": title,
            "authors": authors,
            "summary": summary,
            "keywords": keywords,
            "methods": methods,
            "results": results_text,
            "references": references,
            "funding_sources": funding_sources,
            "conference_info": conference_info,
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time,
                "method": "qwen_local",
                "model": "Qwen2.5-1.5B-Instruct",
                "device": str(next(extractor.model.parameters()).device),
                "text_length": len(text),
                "do_sample": False,
                "max_tokens": 150
            }
        }

        # Display results
        print(f"\n📄 TITLE: {results['title'][:100]}")
        print(f"\n👥 AUTHORS: {len(results['authors'])} found")
        for author in results["authors"]:
            affil_str = f" ({', '.join(author['affiliations'])})" if author['affiliations'] else ""
            print(f"   • {author['name']}{affil_str}")

        print(f"\n📝 SUMMARY: {results['summary'][:100]}...")
        print(f"\n🔑 KEYWORDS: {', '.join(results['keywords'][:5])}")
        print(f"\n🔬 METHODS: {results['methods'][:100]}...")
        print(f"\n📊 RESULTS: {results['results'][:100]}...")
        print(f"\n📚 REFERENCES: {len(results['references'])} found")
        for ref in results['references'][:2]:  # Show first 2
            print(f"   • {ref['title'][:50]}...")
        print(f"\n💰 FUNDING: {len(results['funding_sources'])} sources")
        for funding in results['funding_sources'][:2]:  # Show first 2
            print(f"   • {funding[:50]}...")
        print(f"\n🏛️  CONFERENCE: {results['conference_info']['location']} | {results['conference_info']['date']}")
        print(f"⏱️  Processing time: {results['extraction_metadata']['processing_time']:.2f}s")

        # Save results
        output_path = Path("../output/method2_qwen_results.json")
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved to: {output_path}")
        print("✅ Method 2 completed successfully!")

    except Exception as e:
        print(f"❌ Qwen extraction failed: {e}")
        print("   This may be due to insufficient GPU memory or model download issues")

else:
    print("❌ Test poster not found")


# In[ ]:




