# Enhanced Parsing Fixes v2 - Improved Author Logic

def enhanced_references_cleaning(parsed_refs: str) -> str:
    """Enhanced cleaning for two-step references method"""
    
    # Remove the specific verbose prefix from two-step method
    if "here are the parsed references in the desired format:" in parsed_refs.lower():
        # Split by lines and remove the verbose prefix lines
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
    
    # Remove leading quotes that might be left over
    parsed_refs = parsed_refs.strip('"').strip("'").strip()
    
    return parsed_refs


def enhanced_author_parsing_v2(response: str) -> list:
    """Enhanced author parsing v2 - Better handling of complex nested parentheses"""
    
    import re
    
    # First, let's try a different approach: manually parse by looking for the main " | " separators
    # But ignore " | " that are inside parentheses
    
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
                # Found separator at top level - split here
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
                i += len(separator) - 1  # Skip the separator
            else:
                current_part += char
            
            i += 1
        
        # Add the last part
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
        
        # Extract name and affiliation from "Name (Affiliation)" format
        name = ""
        affiliations = []
        
        if '(' in entry and ')' in entry:
            # Find the main name (everything before the first parenthesis)
            paren_start = entry.find('(')
            name = entry[:paren_start].strip()
            
            # Extract affiliation content (everything inside parentheses)
            affiliation_start = paren_start + 1
            # Find matching closing parenthesis
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
                
                # Split affiliations by " | " (these are legitimate separators within affiliations)
                affiliation_parts = affiliation_text.split(' | ')
                for part in affiliation_parts:
                    part = part.strip()
                    if part and len(part) > 2:
                        affiliations.append(part)
        else:
            # No parentheses, just use the whole entry as name
            name = entry
        
        # Validate that this is actually a person's name
        if not name:
            continue
        
        # Filter out institutional keywords
        institutional_keywords = [
            'department', 'university', 'institute', 'center', 'centre', 
            'school', 'college', 'laboratory', 'lab', 'division',
            'research', 'faculty', 'hospital', 'clinic', 'universitat',
            'catalunya', 'politecnica', 'upc'  # Add specific problematic terms
        ]
        
        name_lower = name.lower()
        
        # Skip if the "name" is clearly an institution or contains institutional keywords
        if any(keyword in name_lower for keyword in institutional_keywords):
            continue
        
        # Skip if it starts with prepositions
        if name_lower.startswith(('of ', 'for ', 'and ', 'the ', 'de ', 'del ')):
            continue
        
        # Skip single words that don't look like names
        name_words = name.split()
        if len(name_words) == 1:
            # Single words must be title case and reasonable length to be names
            if not (name.istitle() and 3 <= len(name) <= 15):
                continue
        
        # Must have at least 2 words for full names, or be a reasonable single name
        if len(name_words) >= 2 or (len(name_words) == 1 and name.istitle() and 3 <= len(name) <= 15):
            authors.append({
                "name": name,
                "affiliations": affiliations,
                "email": None
            })
            
            if len(authors) >= 6:  # Limit to 6 authors
                break
    
    return authors


# Test with the actual problematic response
test_response = 'Merve Gul (University of Pavia | Department of Chemical Engineering, Universitat Politècnica de Catalunya (UPC-EEBE)) | Ida Genta (University of Pavia) | Maria M. Perez Madrigal (Department of Chemical Engineering, Universitat Politècnica de Catalunya) | Carlos Aleman (Universitat Politècnica de Catalunya) | Enrica Chiesa (University of Pavia)'

if __name__ == "__main__":
    print("Testing enhanced author parsing v2:")
    result = enhanced_author_parsing_v2(test_response)
    for i, author in enumerate(result, 1):
        affil_str = f" - Affiliations: {author['affiliations']}" if author['affiliations'] else " - No affiliations"
        print(f"{i}. {author['name']}{affil_str}")
    
    print(f"\nFound {len(result)} authors (should be 5)")
    
    # Check if we got the expected authors
    expected_authors = ["Merve Gul", "Ida Genta", "Maria M. Perez Madrigal", "Carlos Aleman", "Enrica Chiesa"]
    found_names = [author['name'] for author in result]
    
    print("\nExpected vs Found:")
    for expected in expected_authors:
        status = "✅" if expected in found_names else "❌"
        print(f"{status} {expected}")
    
    # Test references cleaning
    test_refs = 'Here are the parsed references in the desired format:\n\n"Front. Bioeng. Biotechnol. - Vega-Váquez, P. et al. (2020) | Biomed. Pharmacother. - Fu, Y. S. et al. (2021)"'
    
    print("\nTesting enhanced references cleaning:")
    cleaned = enhanced_references_cleaning(test_refs)
    print(f"Cleaned: {cleaned}")
