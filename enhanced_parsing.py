"""Enhanced parsing fixes for author and references extraction"""

def enhanced_references_cleaning(parsed_refs: str) -> str:
    """Enhanced cleaning for two-step references method"""
    
    # Remove the specific verbose prefix from two-step method
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
    return parsed_refs


def enhanced_author_parsing(response: str) -> list:
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
