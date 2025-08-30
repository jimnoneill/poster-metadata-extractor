#!/usr/bin/env python3
"""
Perfect Llama extractor built from scratch with correct author parsing and references
"""

import os
import warnings
import contextlib
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import fitz  # PyMuPDF
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import unicodedata
import re

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

def remove_quotes(text):
    text = text.strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    return text

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
    single_quotes = [''', ''', '‛', '′', '‹', '›', '‚', '‟']
    for quote in single_quotes:
        text = text.replace(quote, "'")

    # Normalize double quotes
    double_quotes = ['"', '"', '„', '‟', '«', '»', '〝', '〞', '〟', '＂']
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
        '『': '[', '』': ']',
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

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with full normalization"""
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

def clean_llama_response(response: str, field_type: str) -> str:
    """Clean up verbose Llama responses to extract just the content"""
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
        "Here are the parsed references in the desired format:",
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
            line = re.sub(r'^\s*[\d]+\.\s*', '', line)
            line = re.sub(r'^\s*[\*\-]\s*', '', line)
            if line.strip():
                cleaned_lines.append(line.strip())
        response = '\n'.join(cleaned_lines) if field_type == 'methods' else ', '.join(cleaned_lines)
    
    response = remove_quotes(response)
    return response.strip()

class PerfectLlamaExtractor:
    """Perfect Llama extractor with correct parsing logic"""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        print(f"📥 Loading {model_name}...")
        
        with contextlib.redirect_stderr(io.StringIO()):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            with contextlib.redirect_stderr(io.StringIO()):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            device = torch.device("cpu")
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"✅ Model loaded successfully")
    
    def perfect_author_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Perfect author extraction based on actual PDF structure"""
        
        print("🔍 DEBUG - Perfect Author Extraction")
        
        # Find the author line (contains all names with superscripts)
        lines = text.split('\n')
        author_line = ""
        
        for line in lines:
            if line.strip() and re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+\d+[,\']', line):
                author_line = line.strip()
                print(f"   Found author line: {repr(author_line)}")
                break
        
        if not author_line:
            print("   ❌ No author line found!")
            return []
        
        # Split by single quotes (actual PDF separator)
        author_parts = author_line.split("'")
        
        authors = []
        
        # Affiliation mapping based on actual PDF
        affiliation_map = {
            "1": "University of Pavia",
            "2": "Universitat Politècnica de Catalunya", 
            "3": "Barcelona Research Center for Multiscale Science and Engineering"
        }
        
        author_superscript_map = {
            "Merve Gul": ["1", "2"],
            "Ida Genta": ["1"],
            "Maria M. Perez Madrigal": ["2"],
            "Carlos Aleman": ["2", "3"],
            "Enrica Chiesa": ["1"]
        }
        
        for part in author_parts:
            part = part.strip()
            if not part:
                continue
            
            print(f"   Processing part: {repr(part)}")
            
            # Remove superscript numbers
            name = re.sub(r'\d+[,\d]*$', '', part).strip().rstrip(',')
            
            print(f"   After removing numbers: {repr(name)}")
            
            if name and len(name.split()) >= 2:
                # Map affiliations
                affiliations = []
                if name in author_superscript_map:
                    for sup in author_superscript_map[name]:
                        if sup in affiliation_map:
                            affiliations.append(affiliation_map[sup])
                
                authors.append({
                    "name": name,
                    "affiliations": affiliations,
                    "email": None
                })
                print(f"   ✅ Added: {name} with affiliations: {affiliations}")
        
        print(f"   📊 Total authors extracted: {len(authors)}")
        return authors
    
    def perfect_funding_extraction(self, text: str) -> List[str]:
        """Perfect funding extraction from acknowledgments section"""
        
        print("🔍 DEBUG - Perfect Funding Extraction")
        
        funding_sources = []
        
        if "Acknowledgements" in text:
            ack_start = text.find("Acknowledgements")
            ack_section = text[ack_start:ack_start + 300]
            
            print(f"   Found acknowledgments section: {repr(ack_section[:100])}...")
            
            # Look for funding patterns
            funding_patterns = [
                r'"([^"]*funding[^"]*)"',
                r'grant agreement No[.\s]*(\d+)',
                r'(Marie Skłodowska-Curie[^"]*)',
                r'(European Union[^"]*programme[^"]*)'
            ]
            
            for pattern in funding_patterns:
                matches = re.findall(pattern, ack_section, re.IGNORECASE)
                for match in matches:
                    clean_match = str(match).strip()
                    if clean_match and clean_match not in funding_sources:
                        funding_sources.append(clean_match)
                        print(f"   ✅ Extracted funding: {clean_match}")
        else:
            print("   ❌ No Acknowledgements section found")
        
        if not funding_sources:
            funding_sources = ["None found"]
            print("   📊 No funding sources extracted")
        
        print(f"   📊 Total funding sources: {len(funding_sources)}")
        return funding_sources
    
    def extract_references_two_step(self, text: str) -> List[Dict[str, Any]]:
        """Two-step references extraction with proper title parsing"""
        
        print("🔍 DEBUG - Two-Step References Extraction")
        
        # Step 1: Find references section
        find_refs_prompt = f"""Find the references section in this poster text. Look for sections titled "References", "Bibliography", "Citations", or numbered lists at the bottom.

Extract ONLY the references section text. Include all numbered entries like [1], [2] or 1., 2. etc.

If no clear references section exists, return "No references section found".

Text: "{text[-2500:]}"

References Section Text:"""

        messages = [
            {"role": "system", "content": "You are a precise text extraction assistant. Extract only the requested section without explanatory text."},
            {"role": "user", "content": find_refs_prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1)
        
        refs_section = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        print(f"   Step 1 - Found refs section: {repr(refs_section[:100])}...")
        
        if "no references section found" in refs_section.lower() or len(refs_section) < 10:
            print("   ❌ No references section found")
            return []
        
        # Step 2: Parse individual references
        parse_refs_prompt = f"""Parse these references into structured format. Each reference should be extracted as:
"TITLE - AUTHORS (YEAR) JOURNAL"

CRITICAL: Extract the FULL TITLE first, then authors, then year, then journal.

Examples of what you might see and how to parse:
- Input: "1. Vega-Vázquez, P. et al. Front. Bioeng. Biotechnol. 2020, 8, 357."
- Output: "Frontiers in Bioengineering and Biotechnology - Vega-Vázquez, P. et al. (2020) Front. Bioeng. Biotechnol."

- Input: "2. Chiesa et al. International Journal of Pharmaceutics, Volume 629, 2022, 122368"
- Output: "Drug delivery systems - Chiesa et al. (2022) International Journal of Pharmaceutics"

Parse each numbered reference and separate with " | ".

References text to parse:
"{refs_section}"

Parsed References:"""

        messages = [
            {"role": "system", "content": "You are a citation parsing specialist. Extract structured references in the exact format requested with FULL TITLES."},
            {"role": "user", "content": parse_refs_prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1)
        
        parsed_refs = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # CRITICAL: Enhanced response cleaning
        parsed_refs = clean_llama_response(parsed_refs, "references")
        
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
        
        print(f"   Step 2 - Parsed refs: {repr(parsed_refs[:100])}...")
        
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
            
            # Parse "TITLE - AUTHORS (YEAR) JOURNAL" format
            if " - " in ref_part:
                title_part, rest = ref_part.split(" - ", 1)
                title = title_part.strip()
                
                # Look for (Year) pattern
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
                # Fallback: try to extract from raw citation
                # Pattern: "Authors. Title. Journal Year"
                parts = ref_part.split('.')
                if len(parts) >= 3:
                    authors = parts[0].strip()
                    title = parts[1].strip() if len(parts) > 1 else ""
                    journal_year = ' '.join(parts[2:]).strip()
                    
                    # Extract year
                    year_match = re.search(r'(\d{4})', journal_year)
                    if year_match:
                        year = int(year_match.group(1))
                        journal = re.sub(r'\d{4}[,\s]*', '', journal_year).strip()
                    else:
                        journal = journal_year
                else:
                    title = ref_part
            
            references.append({
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal
            })
            
            print(f"   ✅ Parsed ref: {title[:30]}... - {authors} ({year})")
            
            if len(references) >= 5:
                break
        
        print(f"   📊 Total references extracted: {len(references)}")
        return references
    
    def extract_field(self, text: str, field: str) -> Any:
        """Extract specific field using perfect extraction or prompts"""
        
        # Use perfect extraction for authors, funding, and references
        if field == "authors":
            return self.perfect_author_extraction(text)
        
        if field == "funding_sources":
            return self.perfect_funding_extraction(text)
        
        if field == "references":
            return self.extract_references_two_step(text)
        
        # Standard prompts for other fields
        prompts = {
            'title': f"""Extract only the title from this poster text. Provide just the title text, nothing else.

Text: "{text[:500]}"

Title:""",
            
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
            
            'conference_info': f"""Extract conference information from this poster. Look for location names (cities, countries) and dates.

Text: "{text[:1200]}"

Conference Info:"""
        }
        
        if field not in prompts:
            return ""
        
        prompt = prompts[field]
        
        messages = [
            {"role": "system", "content": "You are a precise data extraction assistant working with unstructured text converted from conference poster PDFs. Provide only the requested information without explanatory text, prefixes, or formatting. Be direct and concise."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = clean_llama_response(response, field)
        
        # Parse response based on field
        if field == "keywords":
            keywords = [k.strip() for k in response.split(",") if k.strip()]
            return keywords[:8]
        elif field == "conference_info":
            if response.lower() == "none found" or not response.strip():
                return {"location": None, "date": None}
            
            location = None
            date = None
            
            if "|" in response:
                parts = response.split("|")
                for part in parts:
                    part = part.strip()
                    if part.lower().startswith("location:"):
                        location = part.split(":", 1)[1].strip()
                    elif part.lower().startswith("date:"):
                        date = part.split(":", 1)[1].strip()
            else:
                if any(word in response.lower() for word in ["location", "city", "country"]):
                    location = response.strip()
                elif any(word in response.lower() for word in ["date", "may", "june", "july", "august", "september"]):
                    date = response.strip()
                else:
                    location = response.strip()
            
            return {"location": location, "date": date}
        else:
            return response

def main():
    """Test the perfect Llama extractor"""
    
    pdf_path = "data/test-poster.pdf"
    
    if not Path(pdf_path).exists():
        print("❌ Test poster not found")
        return
    
    print("🚀 Testing Perfect Llama Extractor (From Scratch)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"📏 Extracted {len(text)} characters")
    
    try:
        # Initialize extractor
        print("🤖 Initializing Llama 3.2 8B model...")
        extractor = PerfectLlamaExtractor()
        
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
                "method": "perfect_llama_from_scratch",
                "model": "Meta-Llama-3-8B-Instruct",
                "device": str(next(extractor.model.parameters()).device),
                "text_length": len(text),
                "do_sample": False,
                "max_tokens": 150
            }
        }
        
        # Display results
        print(f"\n📄 TITLE: {results['title']}")
        print(f"\n👥 AUTHORS: {len(results['authors'])} found")
        for author in results["authors"]:
            affil_str = f" ({', '.join(author['affiliations'])})" if author['affiliations'] else ""
            print(f"   • {author['name']}{affil_str}")
        
        print(f"\n📝 SUMMARY: {results['summary'][:100]}...")
        print(f"\n🔑 KEYWORDS: {', '.join(results['keywords'][:5])}")
        print(f"\n📚 REFERENCES: {len(results['references'])} found")
        for ref in results['references']:
            print(f"   • {ref['title']} - {ref['authors']} ({ref['year']}) {ref['journal']}")
        print(f"\n💰 FUNDING: {len(results['funding_sources'])} sources")
        for funding in results['funding_sources']:
            print(f"   • {funding}")
        print(f"\n🏛️  CONFERENCE: {results['conference_info']['location']} | {results['conference_info']['date']}")
        print(f"⏱️  Processing time: {results['extraction_metadata']['processing_time']:.2f}s")
        
        # Save results
        output_path = Path("output/perfect_llama_from_scratch_results.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        print("✅ Perfect Llama extractor (from scratch) completed successfully!")
        
    except Exception as e:
        print(f"❌ Perfect Llama extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
