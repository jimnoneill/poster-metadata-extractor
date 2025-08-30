#!/usr/bin/env python3
"""
Final targeted fixes for remaining issues:
1. Debug Llama author parsing 
2. Fix funding extraction to focus on acknowledgments section
"""

import json
import re

def debug_llama_author_issue():
    """Debug why Llama enhanced author parsing isn't working"""
    
    # Check if the enhanced method was actually added to Llama notebook
    with open('notebooks/02_method2a_llama32_8b_local.ipynb', 'r') as f:
        notebook = json.load(f)
    
    for i, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source_text = ''.join(cell.get('source', []))
            
            if 'enhanced_author_parsing' in source_text:
                print(f"✅ Enhanced author parsing method found in cell {i}")
                return True
            
            if 'return self.enhanced_author_parsing(response)' in source_text:
                print(f"✅ Enhanced author parsing call found in cell {i}")
                return True
    
    print("❌ Enhanced author parsing method not found in Llama notebook!")
    return False

def apply_targeted_fixes():
    """Apply targeted fixes for the remaining issues"""
    
    notebooks = [
        ('notebooks/02_method2a_llama32_8b_local.ipynb', 'Llama'),
        ('notebooks/02_method2_qwen_local.ipynb', 'Qwen')
    ]
    
    for notebook_path, model_type in notebooks:
        print(f"\nApplying targeted fixes to {model_type} notebook...")
        
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        for i, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source_text = ''.join(cell.get('source', []))
                
                # Find the extractor class
                if f'{model_type}Extractor' in source_text and 'extract_field' in source_text:
                    print(f"Found {model_type}Extractor in cell {i}")
                    
                    # Fix 1: Ensure enhanced author parsing is properly called
                    if 'elif field == "authors":' in source_text and 'enhanced_author_parsing' not in source_text:
                        print(f"Fixing {model_type} author parsing call...")
                        # Replace the author parsing logic
                        author_pattern = r'(elif field == "authors":.*?)(elif field == "keywords":)'
                        replacement = r'''elif field == "authors":
            # Use enhanced parsing to handle complex nested parentheses
            return self.enhanced_author_parsing(response)
            
        \2'''
                        source_text = re.sub(author_pattern, replacement, source_text, flags=re.DOTALL)
                    
                    # Fix 2: Enhanced funding extraction for acknowledgments
                    if "'funding_sources':" in source_text:
                        print(f"Enhancing {model_type} funding extraction...")
                        
                        # Create a much more targeted funding prompt
                        funding_prompt = f'''f"""Extract funding/acknowledgment information from this poster. 

CRITICAL: Look specifically at the BOTTOM/END of the poster text for sections like:
- "Acknowledgments", "Acknowledgements", "Funding", "Support"
- Look for the LAST 800 characters of the text which typically contain acknowledgments

Common patterns in acknowledgments:
• "We acknowledge...", "The authors acknowledge...", "This work was supported by..."
• "Financial support from...", "Funded by...", "Grant support from..."  
• "Thanks to...", "Supported by...", "This research was funded by..."
• Grant numbers: "Grant No. XXXXX", "Project XXXXX", "#XXXXX"
• Funding agencies: "NSF", "NIH", "EU", "Horizon 2020", "ERC", etc.

Focus on the END of the text (acknowledgments are usually at bottom of posters):
Text (last 800 chars): "{{text[-800:]}}"

Full text for context: "{{text}}"

Extract specific funding sources, grant numbers, or agencies. List separated by commas.
If no acknowledgments/funding found, return "None found".

Funding/Acknowledgments:"""'''
                        
                        # Replace the funding prompt
                        funding_pattern = r"('funding_sources': f\"\"\".*?Funding Sources:\"\"\")"
                        source_text = re.sub(funding_pattern, "'funding_sources': " + funding_prompt, source_text, flags=re.DOTALL)
                    
                    # Update the cell
                    cell['source'] = source_text.split('\n')
                    for j in range(len(cell['source'])-1):
                        if not cell['source'][j].endswith('\n'):
                            cell['source'][j] += '\n'
        
        # Save the notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"✅ Applied targeted fixes to {model_type} notebook")

def main():
    """Apply final targeted fixes"""
    
    print("🎯 FINAL TARGETED FIXES")
    print("=" * 50)
    print("Issues to resolve:")
    print("1. Llama missing 2 authors (Merve Gul, Carlos Aleman)")  
    print("2. Both models extracting references instead of acknowledgments")
    print()
    
    # Debug Llama author issue
    print("🔍 Debugging Llama author parsing...")
    if not debug_llama_author_issue():
        print("Need to fix Llama enhanced author parsing method!")
    
    # Apply targeted fixes
    print("\n🔧 Applying targeted fixes...")
    apply_targeted_fixes()
    
    print("\n✅ TARGETED FIXES APPLIED")
    print("\nNext: Run notebooks to test:")
    print("1. Llama should now find all 5 authors")
    print("2. Both should find acknowledgments at bottom of poster")

if __name__ == "__main__":
    main()
