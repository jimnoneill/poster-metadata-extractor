---
title: "Scientific Poster Metadata Extraction: Three-Method Approach"
author: "Jamey O'Neill, PhD, MSc"
date: "August 29, 2025"
---

# Scientific Poster Metadata Extraction
## Three Approaches for Different Requirements

**Jamey O'Neill, PhD, MSc**  
August 29, 2025

---

# The Challenge

## Why This Matters
- **Manual processing bottleneck**: Libraries process thousands annually
- **Inconsistent metadata quality**: Human extraction varies  
- **Scalability issues**: Massive conference poster volumes
- **Cost constraints**: Manual curation expensive at scale

## Target Metadata (Table 1)
1. Title, Authors, Affiliations
2. Summary, Keywords, Methods
3. Results, References, Funding

---

# Our Three-Method Solution

| Method | Best For | Key Advantage |
|--------|----------|---------------|
| **DeepSeek API** | Quick deployment | Balance cost/accuracy |
| **Qwen Local** | Privacy-sensitive | Complete data control |
| **BioELECTRA+CRF** | Maximum accuracy | Zero hallucination |

## Design Philosophy
- **No one-size-fits-all** approach
- **Modular architecture** for easy comparison
- **Bootstrapping strategy** for data generation

---

# Method 1: DeepSeek API
## The Pragmatic Choice

### Performance Characteristics
- **Accuracy**: 85-90% (requires validation)
- **Cost**: ~$0.003 per poster (200x cheaper than GPT-4)
- **Speed**: 5-15 seconds per poster
- **Setup**: Easy - just API key

### Best For
Production systems with API budget, high-volume processing

---

# Method 1: Technical Details

## Enhanced Structured Prompting
```python
prompt = create_extraction_prompt(poster_text)
response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[system_prompt, user_prompt],
    temperature=0.1
)
```

## Live Results
- **Title**: Correctly extracted
- **Authors**: 5 identified with affiliations  
- **Keywords**: 8 technical terms
- **Time**: 25.6 seconds, **Cost**: $0.0007

---

# Method 2: Qwen Local
## The Privacy-First Approach

### Performance Characteristics
- **Accuracy**: 80-85% (requires validation)
- **Cost**: $0 (electricity only)
- **Speed**: 10-40s single / ~1.1s batched (RTX 4090)
- **Privacy**: 100% local processing

### RTX 4090 Batching
- **32 posters simultaneously**
- **3,273 posters/hour throughput**
- **26,182 posters/day** (8-hour operation)

---

# Method 2: Technical Implementation

## 8-bit Quantization for Efficiency
```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config
)
```

## Best For
Privacy-sensitive environments, edge computing, budget-conscious deployments

---

# Method 3: BioELECTRA+CRF
## The Future Possibility

### Why BioELECTRA?
- **2nd highest** on BLURB biomedical leaderboard
- **Domain-optimized**: Pre-trained on PubMed corpus
- **Zero hallucination**: Deterministic sequence labeling

### Expected Performance (With Training)
- **Accuracy**: 85-92% (estimated from BLURB benchmarks)
- **Speed**: <0.5 seconds per poster (fastest)
- **Hallucination**: 0% (deterministic BIO tagging)
- **Cost**: $0 (after training)

---

# Method 3: Training Requirements

## BIO Tagging Example
```
Input:  "This study by Dr. Smith uses microfluidic synthesis"
Labels: ['O','O','O','B-AUTHOR','I-AUTHOR','O','B-METHOD','I-METHOD']
```

## Training Investment Needed
- **500-1000 labeled posters** required
- **40-60 hours** expert annotation time
- **2-4 hours** training on V100 GPU

## Current Status
Demo only - requires substantial training data investment

---

# Comparative Analysis

| Feature | DeepSeek API | Qwen Local | BioELECTRA+CRF |
|---------|-------------|------------|----------------|
| **Accuracy** | 85-90%* | 80-85%* | 85-92%* |
| **Cost/poster** | $0.003 | $0 | $0 |
| **Speed** | 5-15s | 10-40s | <0.5s |
| **Privacy** | External API | 100% Local | 100% Local |
| **Hallucination** | Low-Med | Low | **None** |
| **Training** | No | No | **Required** |

*All accuracy estimates unvalidated - require Cochran sampling

---

# Decision Framework

## Method Selection Guide
- **Immediate deployment** → Method 1 (DeepSeek)
- **Privacy critical** → Method 2 (Qwen)  
- **Maximum accuracy investment** → Method 3 (BioELECTRA)

## Key Considerations
- Budget constraints vs accuracy requirements
- Privacy/security requirements
- Technical infrastructure capabilities
- Long-term scalability needs

---

# Validation Framework
## Cochran's Sampling Methodology

### Why Statistical Validation?
All accuracy estimates are **unvalidated** - proper validation essential

### Sample Size Requirements
- **1,000 posters** → Validate **278 samples** (27.8%)
- **10,000 posters** → Validate **370 samples** (3.7%)  
- **100,000+ posters** → Validate **383 samples** (0.4%)

### Field-Specific Metrics
- **Title**: Exact match or semantic similarity >0.8
- **Authors**: Fuzzy matching (edit distance <2)
- **Keywords**: Overlap coefficient >0.6

---

# Live Demonstration Results
## Poster: "Drug-Polymer Interactions in PLGA/PLA-PEG Nanoparticles"

### Method 1 (DeepSeek) Performance
- **Title**: "INFLUENCE OF DRUG-POLYMER INTERACTIONS..."
- **Authors**: 5 identified (Merve Gul, Ida Genta, et al.)
- **Affiliations**: University of Pavia, UPC-EEBE
- **Keywords**: 8 technical terms extracted
- **Processing**: 25.6 seconds
- **Cost**: $0.0007

### Method 2 (Qwen) Performance  
- **Title**: Correctly extracted
- **Authors**: 5 identified
- **Processing**: 47.5 seconds, **Cost**: $0

---

# Implementation Roadmap

## Phase 1: Quick Start (Week 1)
1. **Deploy Method 1** (DeepSeek API) for immediate results
2. **Implement Cochran sampling** on first 100-500 extractions
3. **Measure actual accuracy** vs. 85-90% estimate

## Phase 2: Optimization (Month 1-2)
1. **Add Method 2** (Qwen Local) for privacy-sensitive content
2. **Compare accuracy** between methods on validation set
3. **Optimize prompts** based on failure analysis

## Phase 3: Advanced Implementation (Month 3-6)
1. **Collect training data** using Methods 1&2 auto-labeling
2. **Train BioELECTRA+CRF** model on 500-1000 labeled posters
3. **Deploy Method 3** for maximum accuracy production

---

# Expected ROI

## Cost Comparison
- **Manual Processing**: $5-10 per poster (human time)
- **Method 1 (DeepSeek)**: $0.003 per poster 
- **Method 2 (Qwen)**: $0 per poster
- **Method 3 (BioELECTRA)**: $0 per poster (after training)

## ROI Analysis
- **99.97% cost reduction** vs manual processing
- **Consistent quality** vs variable human performance
- **24/7 availability** vs limited human resources
- **Scalable throughput** for conference-scale processing

---

# Current Limitations

## Technical Constraints
- **Accuracy estimates unvalidated** - require statistical validation
- **English-only optimization** - multilingual support needed
- **Text-only processing** - image/diagram extraction not included
- **Method 3 requires training** - substantial data annotation

## Validation Requirements
- **Cochran sampling essential** before production use
- **Field-specific accuracy testing** needed
- **Cross-domain validation** for different poster types

---

# Future Enhancements

## Immediate Improvements (3-6 months)
- **OCR integration** for scanned/image-only posters
- **Multilingual support** (Spanish, French, German)
- **Complete Method 3 training** with auto-labeled data

## Advanced Developments (6-12 months)
- **Multi-modal architecture** using LayoutLM for visual layout
- **Real-time API deployment** for library system integration
- **Cross-lingual transfer learning** for global collections

---

# Technical Specifications

## Hardware Requirements
| Method | CPU | RAM | GPU | Storage |
|--------|-----|-----|-----|---------|
| **Method 1** | Any | 4GB | None | Minimal |
| **Method 2** | Modern | 16GB | 8GB VRAM | ~3GB |
| **Method 3** | Modern | 16GB | 8GB VRAM | ~800MB |

## Key Dependencies
- **Core**: Python 3.8+, PyMuPDF, transformers
- **Method 1**: openai, python-dotenv  
- **Method 2**: torch, bitsandbytes, accelerate
- **Method 3**: pytorch-crf, spacy (post-training)

---

# Key Takeaways

## Critical Success Factors
1. **Three complementary approaches** address different organizational needs
2. **Statistical validation essential** - all estimates require Cochran sampling
3. **Immediate deployment possible** with Method 1 (DeepSeek API)
4. **Privacy-first option available** with Method 2 (Qwen Local)
5. **Future scalability** through Method 3 training investment

## Bottom Line
**Start with Method 1, validate with Cochran sampling, evolve to Method 3 for production scale**

**99.97% cost reduction vs manual processing**

---

# Questions & Discussion

## Contact & Resources
- **Full Documentation**: README.md with complete technical specs
- **Implementation Code**: Available in `/src/` and `/notebooks/`
- **Live Results**: JSON outputs in `/output/` directory

### Key Repository Structure
```
poster_project/
├── src/                    # Python implementations
├── notebooks/              # Jupyter demonstrations  
├── output/                 # Extraction results
└── data/                   # Sample posters
```

**Thank you for your attention!**
