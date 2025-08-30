# Enhanced Parsing Fixes for Llama Author and References Issues

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


def enhanced_author_parsing(response: str) -> list:
    """Enhanced author parsing to handle nested parentheses and complex affiliations"""
    
    # The response format: "Author (Affiliation | Complex Affiliation (Sub-part)) | Author (Affiliation)"
    # Problem: Simple split on "|" doesn't work due to nested "|" in affiliations
    
    authors = []
    
    # Strategy: Use a smarter parsing approach
    # Find author names by looking for patterns: "Name (" at the start of entries
    
    import re
    
    # Split by " | " but be careful about nested parentheses  
    # Use regex to find author entries: "Name (affiliation)" patterns
    
    # Pattern: Name (potentially with spaces) followed by opening parenthesis
    author_pattern = r'([A-Z][a-zA-Z\s\.]+?)\s*\('
    
    # Find all potential author starts
    author_starts = list(re.finditer(author_pattern, response))
    
    for i, match in enumerate(author_starts):
        author_name = match.group(1).strip()
        start_pos = match.start()
        
        # Find the end of this author's entry (next author start or end of string)
        if i < len(author_starts) - 1:
            end_pos = author_starts[i + 1].start()
        else:
            end_pos = len(response)
        
        author_section = response[start_pos:end_pos].strip()
        
        # Extract affiliation by finding content between parentheses
        # Handle nested parentheses properly
        affiliations = []
        if '(' in author_section:
            # Find the main affiliation content
            paren_start = author_section.find('(')
            # Find the matching closing parenthesis (handling nesting)
            paren_count = 0
            paren_end = -1
            for j, char in enumerate(author_section[paren_start:], paren_start):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        paren_end = j
                        break
            
            if paren_end > paren_start:
                affiliation_text = author_section[paren_start+1:paren_end]
                # Split by " | " for multiple affiliations, but be careful
                affiliation_parts = affiliation_text.split(' | ')
                for part in affiliation_parts:
                    part = part.strip()
                    if part and len(part) > 3:  # Valid affiliation
                        affiliations.append(part)
        
        # Filter out non-author names (departments, institutions, etc.)
        institutional_keywords = [
            'department', 'university', 'institute', 'center', 'centre', 
            'school', 'college', 'laboratory', 'lab', 'division',
            'research', 'faculty', 'hospital', 'clinic'
        ]
        
        name_lower = author_name.lower()
        
        # Skip if the "name" is clearly an institution
        if any(keyword in name_lower for keyword in institutional_keywords):
            continue
        
        # Skip if it starts with prepositions or looks like an affiliation
        if name_lower.startswith(('of ', 'for ', 'and ', 'the ')):
            continue
        
        # Require at least first and last name (or accept single names if they look right)
        if author_name and (len(author_name.split()) >= 2 or 
                          (len(author_name.split()) == 1 and len(author_name) >= 3 and author_name.istitle())):
            authors.append({
                "name": author_name,
                "affiliations": affiliations,
                "email": None
            })
            
            if len(authors) >= 6:  # Limit to 6 authors
                break
    
    return authors


# Test the parsing with the actual problematic response
test_response = 'Merve Gul (University of Pavia | Department of Chemical Engineering, Universitat Politècnica de Catalunya (UPC-EEBE)) | Ida Genta (University of Pavia) | Maria M. Perez Madrigal (Department of Chemical Engineering, Universitat Politècnica de Catalunya) | Carlos Aleman (Universitat Politècnica de Catalunya) | Enrica Chiesa (University of Pavia)'

if __name__ == "__main__":
    print("Testing enhanced author parsing:")
    result = enhanced_author_parsing(test_response)
    for i, author in enumerate(result, 1):
        print(f"{i}. {author['name']} - Affiliations: {author['affiliations']}")
    
    print(f"\nFound {len(result)} authors (should be 5)")
    
    # Test references cleaning
    test_refs = 'Here are the parsed references in the desired format:\n\n"Front. Bioeng. Biotechnol. - Vega-Váquez, P. et al. (2020) | Biomed. Pharmacother. - Fu, Y. S. et al. (2021)"'
    
    print("\nTesting enhanced references cleaning:")
    cleaned = enhanced_references_cleaning(test_refs)
    print(f"Cleaned: {cleaned}")
