#!/usr/bin/env python3
"""
Fix the notebooks with enhanced parsing and funding extraction
"""

import json
import re

def enhanced_author_parsing_code():
    """Return the enhanced author parsing method as a string"""
    return '''
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
    '''

def enhanced_funding_prompt():
    """Return enhanced funding extraction prompt"""
    return '''f"""Extract funding information from this poster text. Look specifically for ACKNOWLEDGMENTS/ACKNOWLEDGEMENTS sections (usually at bottom of poster).

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

Funding/Acknowledgments:"""'''

def apply_fixes_to_notebook(notebook_path: str, model_type: str):
    """Apply all fixes to a specific notebook"""
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    print(f"Applying fixes to {notebook_path} ({model_type})...")
    
    # Find and modify the extractor class cell
    for i, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            source_text = ''.join(source_lines)
            
            # Check if this is the extractor class
            if f'{model_type}Extractor' in source_text and 'extract_field' in source_text:
                print(f"Found {model_type}Extractor class in cell {i}")
                
                # 1. Fix references cleaning in two-step method
                if 'extract_references_two_step' in source_text:
                    print("Fixing references cleaning...")
                    # Add enhanced cleaning after the decode step
                    old_pattern = r'(parsed_refs = self\.tokenizer\.decode.*?\.strip\(\))'
                    replacement = r'''\1
        
        # CRITICAL: Enhanced response cleaning for two-step method
        parsed_refs = clean_''' + model_type.lower() + r'''_response(parsed_refs, "references")
        
        # Additional cleaning for verbose prefixes
        if "here are the parsed references in the desired format:" in parsed_refs.lower():
            lines = parsed_refs.split('\\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if (line and 
                    not line.lower().startswith('here are the parsed references') and
                    not line.lower().startswith('here are') and
                    line != ''):
                    cleaned_lines.append(line)
            parsed_refs = ' '.join(cleaned_lines)
        
        parsed_refs = parsed_refs.strip('"').strip("'").strip()'''
                    
                    source_text = re.sub(old_pattern, replacement, source_text)
                
                # 2. Add enhanced author parsing method
                if 'def extract_references_two_step' in source_text:
                    print("Adding enhanced author parsing method...")
                    # Insert enhanced author parsing method before extract_field
                    insertion_point = source_text.find('def extract_field(')
                    if insertion_point > 0:
                        enhanced_method = enhanced_author_parsing_code()
                        source_text = (source_text[:insertion_point] + 
                                     enhanced_method + '\n    ' + 
                                     source_text[insertion_point:])
                
                # 3. Replace author parsing logic
                if 'elif field == "authors":' in source_text:
                    print("Replacing author parsing logic...")
                    # Replace the entire author parsing section
                    old_authors_pattern = r'(elif field == "authors":)(.*?)(elif field == "keywords":)'
                    replacement = r'''\1
            # Use enhanced parsing to handle complex nested parentheses
            return self.enhanced_author_parsing(response)
            
        \3'''
                    source_text = re.sub(old_authors_pattern, replacement, source_text, flags=re.DOTALL)
                
                # 4. Enhanced funding extraction prompt
                if "'funding_sources':" in source_text:
                    print("Enhancing funding extraction prompt...")
                    old_funding_pattern = r"('funding_sources': f\"\"\".*?Funding Sources:\"\"\")"
                    enhanced_funding = enhanced_funding_prompt()
                    replacement = "'funding_sources': " + enhanced_funding
                    source_text = re.sub(old_funding_pattern, replacement, source_text, flags=re.DOTALL)
                
                # Update the cell source
                cell['source'] = source_text.split('\n')
                # Ensure proper line endings
                for j in range(len(cell['source'])-1):
                    if not cell['source'][j].endswith('\n'):
                        cell['source'][j] += '\n'
                
                print(f"✅ Applied all fixes to {model_type} notebook")
                break
    
    # Save the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Saved updated {notebook_path}")


def main():
    """Apply fixes to both notebooks"""
    
    notebooks = [
        ('notebooks/02_method2a_llama32_8b_local.ipynb', 'Llama'),
        ('notebooks/02_method2_qwen_local.ipynb', 'Qwen')
    ]
    
    print("🔧 Applying comprehensive fixes to notebooks...")
    print("Fixes include:")
    print("1. Enhanced references cleaning for two-step method")
    print("2. Advanced author parsing with nested parentheses support")
    print("3. Enhanced funding extraction for acknowledgments sections")
    print()
    
    for notebook_path, model_type in notebooks:
        try:
            apply_fixes_to_notebook(notebook_path, model_type)
            print()
        except Exception as e:
            print(f"❌ Error fixing {notebook_path}: {e}")
            print()
    
    print("🎯 All fixes applied! Ready to test the enhanced notebooks.")
    print("\nNext steps:")
    print("1. Run both notebooks to test the fixes")
    print("2. Check funding extraction from acknowledgments")
    print("3. Verify all 5 authors are extracted correctly")
    print("4. Confirm references are clean without verbose prefixes")


if __name__ == "__main__":
    main()
