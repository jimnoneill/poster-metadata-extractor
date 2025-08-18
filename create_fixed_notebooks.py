#!/usr/bin/env python3
"""
Script to create fixed versions of all notebooks with proper API key loading and error handling
"""

import json
import subprocess
import sys
from pathlib import Path

def create_method1_notebook():
    """Create Method 1 notebook with dotenv fix"""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Method 1: DeepSeek API Extraction\\n\\n## Overview\\nCost-effective poster metadata extraction using DeepSeek API. This approach offers the best balance of accuracy and cost for most use cases.\\n\\n## Accuracy Note\\nThe 85-90% accuracy estimate is unvalidated - based on limited testing only. Actual accuracy must be determined through proper Cochran sampling validation before production use.\\n\\n## Performance Characteristics\\n- **Estimated Accuracy**: 85-90% (unvalidated - requires Cochran sampling validation)\\n- **Cost**: ~$0.003 per poster (200x cheaper than GPT-4)\\n- **Speed**: 5-15 seconds per poster\\n- **Hallucination Risk**: Low-Medium (mitigated by structured prompts)\\n- **Setup**: Easy - just requires API key\\n\\n## Best For\\n- Production systems with budget constraints\\n- High-volume processing\\n- Quick prototyping and development"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Imports and setup\\nimport os\\nimport json\\nimport fitz  # PyMuPDF\\nfrom pathlib import Path\\nfrom datetime import datetime\\nfrom typing import Dict, List, Optional, Any\\nimport time\\nfrom openai import OpenAI\\nimport warnings\\nfrom dotenv import load_dotenv\\nwarnings.filterwarnings('ignore')\\n\\n# Load environment variables from .env file\\nload_dotenv()\\n\\nprint(\\\"✅ Imports successful\\\")\\nprint(\\\"🎯 Method 1: DeepSeek API Extraction\\\")"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Read the full script content from method1_deepseek_api.py
    with open('scripts/method1_deepseek_api.py', 'r') as f:
        script_content = f.read()
    
    # Extract function definitions and main execution
    lines = script_content.split('\n')
    
    # Find configuration section
    config_start = None
    config_end = None
    for i, line in enumerate(lines):
        if '# Configuration' in line:
            config_start = i
        if config_start and 'print("✅ Configuration and functions defined")' in line:
            config_end = i + 1
            break
    
    if config_start and config_end:
        config_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": '\n'.join(lines[config_start:config_end])
        }
        nb["cells"].append(config_cell)
    
    # Find extraction functions section
    func_start = None
    func_end = None
    for i, line in enumerate(lines):
        if 'def extract_with_deepseek(' in line:
            func_start = i
        if func_start and 'print("✅ Extraction functions defined")' in line:
            func_end = i + 1
            break
    
    if func_start and func_end:
        func_cell = {
            "cell_type": "code", 
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": '\n'.join(lines[func_start:func_end])
        }
        nb["cells"].append(func_cell)
    
    # Find main execution section
    main_start = None
    for i, line in enumerate(lines):
        if 'if __name__ == "__main__":' in line:
            main_start = i + 1
            break
        elif '# Run extraction' in line:
            main_start = i
            break
    
    if main_start:
        main_lines = []
        for line in lines[main_start:]:
            if line.strip() and not line.startswith('#!/'):
                main_lines.append(line)
        
        main_cell = {
            "cell_type": "code",
            "execution_count": None, 
            "metadata": {},
            "outputs": [],
            "source": '\n'.join(main_lines)
        }
        nb["cells"].append(main_cell)
    
    return nb

def create_method2_notebook():
    """Create Method 2 notebook with AttributeError fixes"""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Method 2: Qwen Local Extraction\\n\\n## Overview\\nLocal small language model for cost-effective poster metadata extraction. Runs entirely on your hardware without API dependencies.\\n\\n## Accuracy Note\\nThe 80-85% accuracy estimate is unvalidated - based on limited testing only. Actual accuracy must be determined through proper Cochran sampling validation before production use.\\n\\n## Performance Characteristics\\n- **Estimated Accuracy**: 80-85% (unvalidated - requires Cochran sampling validation)\\n- **Cost**: $0 (runs locally, only electricity costs)\\n- **Speed**: 10-40 seconds per poster (single), ~1.1s per poster (RTX 4090 batched)\\n- **Hallucination Risk**: Low (structured prompting)\\n- **Setup**: Medium - requires model download and GPU memory\\n\\n## RTX 4090 Batching Capacity\\n- **Recommended batch size**: 32 posters simultaneously\\n- **Throughput**: ~3,273 posters/hour, ~26,182 posters/day (8hrs)\\n\\n## Best For\\n- Privacy-sensitive environments\\n- Budget-conscious deployments\\n- Edge computing scenarios\\n- Development and experimentation"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Read the full script content
    with open('scripts/method2_qwen_local.py', 'r') as f:
        script_content = f.read()
    
    # Split into logical sections and create cells
    lines = script_content.split('\n')
    
    # Imports section
    import_start = 0
    import_end = None
    for i, line in enumerate(lines):
        if 'def extract_text_from_pdf(' in line:
            import_end = i
            break
    
    if import_end:
        import_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": '\n'.join(lines[import_start:import_end])
        }
        nb["cells"].append(import_cell)
    
    # Functions section  
    func_start = import_end
    func_end = None
    for i in range(func_start, len(lines)):
        if 'if __name__ == "__main__":' in lines[i] or '# Run extraction' in lines[i]:
            func_end = i
            break
    
    if func_start and func_end:
        func_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": '\n'.join(lines[func_start:func_end])
        }
        nb["cells"].append(func_cell)
    
    # Main execution
    if func_end:
        main_lines = []
        for line in lines[func_end:]:
            if line.strip() and not line.startswith('#!/'):
                main_lines.append(line)
        
        main_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": '\n'.join(main_lines)
        }
        nb["cells"].append(main_cell)
    
    return nb

def main():
    """Create and execute fixed notebooks"""
    print("🔧 Creating fixed notebooks...")
    
    # Create Method 1 notebook
    nb1 = create_method1_notebook()
    with open('notebooks/01_method1_deepseek_api.ipynb', 'w') as f:
        json.dump(nb1, f, indent=2)
    print("✅ Created fixed Method 1 notebook")
    
    # Create Method 2 notebook  
    nb2 = create_method2_notebook()
    with open('notebooks/02_method2_qwen_local.ipynb', 'w') as f:
        json.dump(nb2, f, indent=2)
    print("✅ Created fixed Method 2 notebook")
    
    print("\n🚀 Executing notebooks...")
    
    # Execute Method 1
    try:
        result = subprocess.run([
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
            'notebooks/01_method1_deepseek_api.ipynb', '--inplace'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Method 1 notebook executed successfully")
        else:
            print(f"❌ Method 1 execution failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Method 1 execution error: {e}")
    
    # Execute Method 2
    try:
        result = subprocess.run([
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            'notebooks/02_method2_qwen_local.ipynb', '--inplace'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Method 2 notebook executed successfully")
        else:
            print(f"❌ Method 2 execution failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Method 2 execution error: {e}")
    
    # Execute Method 3
    try:
        result = subprocess.run([
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            'notebooks/03_method3_bioelectra_crf_demo.ipynb', '--inplace'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Method 3 notebook executed successfully")
        else:
            print(f"❌ Method 3 execution failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Method 3 execution error: {e}")
    
    print("\n🎉 Fixed notebooks created and executed!")

if __name__ == "__main__":
    main()
