"""Enhanced funding extraction specifically for acknowledgments sections"""

def enhanced_funding_extraction_prompt(text: str) -> str:
    """Generate enhanced funding extraction prompt for acknowledgments"""
    
    return f"""Extract funding information from this poster text. Look specifically for:

ACKNOWLEDGMENTS/ACKNOWLEDGEMENTS sections (usually at bottom of poster)
Common funding patterns:
- "We acknowledge...", "The authors acknowledge...", "This work was supported by..."
- "Financial support from...", "Funded by...", "Grant support from..."
- Grant numbers: "Grant No. XXXXX", "Project #XXXXX", "#XXXXX"
- Funding agencies: "NSF", "NIH", "EU", "Horizon 2020", "ERC", "EPSRC", etc.
- University/institutional funding
- Fellowship acknowledgments

Look for text sections with words like:
- "acknowledge", "support", "funding", "grant", "fellowship"
- "financial", "sponsored", "contract", "award"

Extract specific funding sources, grant numbers, or agencies. If found, list them separated by commas.
If no funding information is found, return "None found".

Text: "{text[-2000:]}"

Funding/Acknowledgments:"""


def test_funding_extraction():
    """Test funding extraction with sample acknowledgment text"""
    
    # Sample text simulating poster acknowledgments
    sample_text = """
    CONCLUSION
    • CUDC loaded PLGA NPs were better candidates for controlled drug delivery due
    to hydrophobic interactions along drug and polymer
    • No or slight cytotoxic effect of CUDC NPs were observed on HepG2 cells,
    suggesting biocompatible characteristics along drug and polymer interactions
    
    ACKNOWLEDGMENTS
    We acknowledge financial support from the European Union's Horizon 2020 
    research program under grant agreement No. 123456789. This work was also
    supported by NSF Grant #CHE-2021234 and University Research Fellowship.
    The authors thank the Barcelona Research Center for technical assistance.
    
    REFERENCES
    [1] Vega-Vázquez, P. et al. Front. Bioeng. Biotechnol. 2020, 8, 357.
    """
    
    # Test the enhanced prompt
    prompt = enhanced_funding_extraction_prompt(sample_text)
    print("Enhanced Funding Extraction Prompt:")
    print("=" * 60)
    print(prompt)
    print("\n" + "=" * 60)
    
    # Expected extraction:
    expected = [
        "European Union's Horizon 2020 research program under grant agreement No. 123456789",
        "NSF Grant #CHE-2021234", 
        "University Research Fellowship"
    ]
    
    print(f"Expected extractions: {expected}")


if __name__ == "__main__":
    test_funding_extraction()
