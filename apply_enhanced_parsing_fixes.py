#!/usr/bin/env python3
"""
Apply enhanced parsing fixes to Llama and Qwen notebooks
This script modifies the notebook source code to include the enhanced parsing logic
"""

import json
import re

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


def enhanced_author_parsing_logic():
    """Return the enhanced author parsing code as a string"""
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


def update_notebook_with_fixes(notebook_path: str) -> bool:
    """Update notebook with enhanced parsing fixes"""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Find the cell that contains the extractor class
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source_lines = cell.get('source', [])
                source_text = ''.join(source_lines)
                
                # Check if this is the extractor class definition cell
                if 'class' in source_text and 'Extractor' in source_text and 'extract_field' in source_text:
                    print(f"Found extractor class in {notebook_path}")
                    
                    # Add enhanced author parsing method after the extract_references_two_step method
                    if 'extract_references_two_step' in source_text:
                        # Insert the enhanced parsing method
                        enhanced_method = enhanced_author_parsing_logic()
                        
                        # Find the right place to insert (after extract_references_two_step)
                        insertion_point = source_text.find('def extract_field(')
                        if insertion_point > 0:
                            new_source = (source_text[:insertion_point] + 
                                        enhanced_method + '\n    ' +
                                        source_text[insertion_point:])
                            
                            # Update the source
                            cell['source'] = new_source.split('\n')
                            # Ensure each line ends with \n except the last
                            for i in range(len(cell['source'])-1):
                                if not cell['source'][i].endswith('\n'):
                                    cell['source'][i] += '\n'
                            
                            print(f"✅ Enhanced parsing method added to {notebook_path}")
                    
                    # Update author parsing logic to use the enhanced method
                    if 'elif field == "authors":' in source_text:
                        # Replace the old parsing logic with a call to enhanced method
                        old_authors_section = re.search(
                            r'elif field == "authors":(.*?)elif field == "keywords":',
                            source_text, 
                            re.DOTALL
                        )
                        if old_authors_section:
                            replacement = '''elif field == "authors":
            # Use enhanced parsing to handle complex nested parentheses
            return self.enhanced_author_parsing(response)
            
        '''
                            new_source = source_text.replace(old_authors_section.group(0), 
                                                           replacement + 'elif field == "keywords":')
                            cell['source'] = new_source.split('\n')
                            # Ensure proper line endings
                            for i in range(len(cell['source'])-1):
                                if not cell['source'][i].endswith('\n'):
                                    cell['source'][i] += '\n'
                            
                            print(f"✅ Author parsing updated in {notebook_path}")
                    
                    # Add enhanced references cleaning to the two-step method
                    if 'parsed_refs = self.tokenizer.decode' in source_text:
                        # Add the cleaning logic after the decode step
                        decode_pattern = r'(parsed_refs = self\.tokenizer\.decode.*?\.strip\(\))'
                        
                        def add_cleaning(match):
                            return match.group(1) + '''
        
        # CRITICAL: Apply enhanced response cleaning
        parsed_refs = clean_llama_response(parsed_refs, "references")
        
        # Additional cleaning for two-step method verbose prefixes
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
                        
                        new_source = re.sub(decode_pattern, add_cleaning, source_text, flags=re.DOTALL)
                        if new_source != source_text:
                            cell['source'] = new_source.split('\n')
                            # Ensure proper line endings
                            for i in range(len(cell['source'])-1):
                                if not cell['source'][i].endswith('\n'):
                                    cell['source'][i] += '\n'
                            print(f"✅ Enhanced references cleaning added to {notebook_path}")
                    
                    break
        
        # Save the updated notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating {notebook_path}: {e}")
        return False


def main():
    """Apply enhanced parsing fixes to both notebooks"""
    
    notebooks = [
        'notebooks/02_method2a_llama32_8b_local.ipynb',
        'notebooks/02_method2_qwen_local.ipynb'
    ]
    
    print("🔧 Applying enhanced parsing fixes to notebooks...")
    
    success_count = 0
    for notebook_path in notebooks:
        print(f"\n📝 Processing {notebook_path}...")
        if update_notebook_with_fixes(notebook_path):
            success_count += 1
            print(f"✅ Successfully updated {notebook_path}")
        else:
            print(f"❌ Failed to update {notebook_path}")
    
    print(f"\n🎯 Enhanced parsing fixes applied to {success_count}/{len(notebooks)} notebooks")
    
    if success_count == len(notebooks):
        print("🚀 All notebooks successfully enhanced! Ready for testing.")
    else:
        print("⚠️ Some notebooks failed to update. Manual fixes may be needed.")


if __name__ == "__main__":
    main()
