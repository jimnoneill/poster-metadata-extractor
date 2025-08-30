#!/usr/bin/env python3
"""Quick debug of author extraction issue"""

import re

def debug_author_extraction():
    """Debug why the perfect author extraction is failing"""
    
    # The actual author line from PDF
    author_line = "Merve Gul1,2'Ida Genta1'Maria M. Perez Madrigal2'Carlos Aleman2,3'Enrica Chiesa1"
    
    print("🔍 DEBUGGING AUTHOR EXTRACTION")
    print(f"Author line: {repr(author_line)}")
    
    # Split by single quotes
    parts = author_line.split("'")
    print(f"After split by quotes: {parts}")
    
    authors = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        print(f"Processing part: {repr(part)}")
        
        # Remove superscript numbers
        name = re.sub(r'\d+[,\d]*$', '', part).strip().rstrip(',')
        print(f"After removing numbers: {repr(name)}")
        
        if name and len(name.split()) >= 2:
            authors.append(name)
            print(f"✅ Added: {name}")
        else:
            print(f"❌ Rejected: {name} (too short or not enough words)")
    
    print(f"\nFinal authors: {authors}")
    print(f"Count: {len(authors)}")
    return authors

if __name__ == "__main__":
    debug_author_extraction()
