#!/usr/bin/env python3
"""
Debug the actual PDF text structure to understand why funding extraction is failing
"""

import fitz  # PyMuPDF
import unicodedata
import re

def remove_quotes(text):
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    else:
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

def debug_pdf_structure():
    """Debug the actual PDF text structure"""
    
    pdf_path = "data/test-poster.pdf"
    
    print("🔍 DEBUGGING PDF TEXT STRUCTURE")
    print("=" * 60)
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    print(f"📏 Total text length: {len(text)} characters")
    print()
    
    # Show the first 500 characters (beginning)
    print("📄 FIRST 500 CHARACTERS (Beginning of poster):")
    print("-" * 50)
    print(repr(text[:500]))
    print()
    
    # Show the last 1000 characters (where acknowledgments should be)
    print("📄 LAST 1000 CHARACTERS (Where acknowledgments should be):")
    print("-" * 50)
    print(repr(text[-1000:]))
    print()
    
    # Look for acknowledgment keywords
    acknowledgment_keywords = [
        'acknowledgment', 'acknowledgments', 'acknowledgement', 'acknowledgements',
        'funding', 'support', 'grant', 'financial', 'sponsored', 'thank'
    ]
    
    print("🔍 SEARCHING FOR ACKNOWLEDGMENT KEYWORDS:")
    print("-" * 50)
    
    text_lower = text.lower()
    
    for keyword in acknowledgment_keywords:
        if keyword in text_lower:
            # Find all occurrences
            positions = []
            start = 0
            while True:
                pos = text_lower.find(keyword, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            
            print(f"✅ '{keyword}' found at positions: {positions}")
            
            # Show context around first occurrence
            if positions:
                pos = positions[0]
                context_start = max(0, pos - 100)
                context_end = min(len(text), pos + 200)
                context = text[context_start:context_end]
                print(f"   Context: ...{repr(context)}...")
        else:
            print(f"❌ '{keyword}' NOT found")
    
    print()
    
    # Look for author names in the text
    print("👥 SEARCHING FOR AUTHOR NAMES:")
    print("-" * 50)
    
    expected_authors = ["Merve Gul", "Ida Genta", "Maria M. Perez Madrigal", "Carlos Aleman", "Enrica Chiesa"]
    
    for author in expected_authors:
        if author.lower() in text_lower:
            pos = text_lower.find(author.lower())
            context_start = max(0, pos - 100)
            context_end = min(len(text), pos + 150)
            context = text[context_start:context_end]
            print(f"✅ '{author}' found at position {pos}")
            print(f"   Context: ...{repr(context)}...")
            print()
        else:
            print(f"❌ '{author}' NOT found")
    
    print()
    
    # Look for reference patterns
    print("📚 SEARCHING FOR REFERENCE PATTERNS:")
    print("-" * 50)
    
    # Look for numbered references
    ref_patterns = [r'\[\d+\]', r'\(\d+\)', r'^\d+\.', r'Front\. Bioeng', r'Bioeng\. Biotechnol']
    
    for pattern in ref_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            print(f"✅ Pattern '{pattern}' found: {matches[:5]}...")
        else:
            print(f"❌ Pattern '{pattern}' NOT found")
    
    return text

if __name__ == "__main__":
    extracted_text = debug_pdf_structure()
    
    # Save the full text for inspection
    with open("debug_full_pdf_text.txt", "w") as f:
        f.write(extracted_text)
    
    print("\n💾 Full text saved to 'debug_full_pdf_text.txt' for manual inspection")
    print("\n🎯 NEXT STEPS:")
    print("1. Examine the saved text file to see exact structure")  
    print("2. Look for patterns in acknowledgments section")
    print("3. Debug why models miss certain author names")
    print("4. Fix the parsing logic based on actual text structure")
