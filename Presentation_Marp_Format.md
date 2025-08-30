---
marp: true
theme: default
paginate: true
backgroundColor: white
header: 'Scientific Poster Metadata Extraction'
footer: 'Jamey O\'Neill, PhD, MSc - August 29, 2025'
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

# Method 1: Live Results

## Enhanced Structured Prompting
```python
response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    temperature=0.1
)
```

## Performance
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
- **Speed**: 10-40s single / 1.1s batched (RTX 4090)
- **Privacy**: 100% local processing

### RTX 4090 Batching
- **32 posters simultaneously**
- **3,273 posters/hour throughput**
- **26,182 posters/day** (8-hour operation)

---

# Method 3: BioELECTRA+CRF
## The Future Possibility

### Why BioELECTRA?
- **2nd highest** on BLURB biomedical leaderboard
- **Domain-optimized**: Pre-trained on PubMed corpus
- **Zero hallucination**: Deterministic sequence labeling

### Expected Performance (With Training)
- **Accuracy**: 85-92% (estimated)
- **Speed**: <0.5 seconds per poster (fastest)
- **Cost**: $0 (after training)

---

# Comparative Analysis

| Feature | DeepSeek API | Qwen Local | BioELECTRA+CRF |
|---------|-------------|------------|----------------|
| **Accuracy** | 85-90%* | 80-85%* | 85-92%* |
| **Cost/poster** | $0.003 | $0 | $0 |
| **Speed** | 5-15s | 10-40s | <0.5s |
| **Privacy** | External API | 100% Local | 100% Local |
| **Training** | No | No | **Required** |

*All accuracy estimates unvalidated

---

# Decision Framework

## Method Selection Guide
- **Immediate deployment** → Method 1 (DeepSeek)
- **Privacy critical** → Method 2 (Qwen)  
- **Maximum accuracy investment** → Method 3 (BioELECTRA)

## Key Considerations
- Budget vs accuracy requirements
- Privacy/security needs
- Technical infrastructure
- Long-term scalability

---

# Validation Framework
## Cochran's Sampling Methodology

### Why Statistical Validation?
All accuracy estimates are **unvalidated** - proper validation essential

### Sample Size Requirements
- **1,000 posters** → Validate **278 samples** (27.8%)
- **10,000 posters** → Validate **370 samples** (3.7%)  
- **100,000+ posters** → Validate **383 samples** (0.4%)

---

# Live Demonstration Results
## Poster: "Drug-Polymer Interactions in PLGA/PLA-PEG Nanoparticles"

### Method 1 (DeepSeek) Performance
- **Title**: "INFLUENCE OF DRUG-POLYMER INTERACTIONS..."
- **Authors**: 5 identified (Merve Gul, Ida Genta, et al.)
- **Processing**: 25.6 seconds, **Cost**: $0.0007

### Method 2 (Qwen) Performance  
- **Title**: Correctly extracted
- **Authors**: 5 identified
- **Processing**: 47.5 seconds, **Cost**: $0

---

# Implementation Roadmap

## Phase 1: Quick Start (Week 1)
1. **Deploy Method 1** (DeepSeek API)
2. **Implement Cochran sampling** 
3. **Measure actual accuracy**

## Phase 2: Optimization (Month 1-2)
1. **Add Method 2** (Qwen Local)
2. **Compare accuracy** between methods
3. **Optimize prompts**

## Phase 3: Advanced Implementation (Month 3-6)
1. **Collect training data** via auto-labeling
2. **Train BioELECTRA+CRF** model
3. **Deploy Method 3** for production

---

# Expected ROI

## Cost Comparison
- **Manual Processing**: $5-10 per poster
- **Method 1 (DeepSeek)**: $0.003 per poster 
- **Method 2 (Qwen)**: $0 per poster
- **Method 3 (BioELECTRA)**: $0 per poster (after training)

## ROI Analysis
- **99.97% cost reduction** vs manual processing
- **Consistent quality** vs variable human performance
- **24/7 availability** vs limited human resources

---

# Technical Specifications

## Hardware Requirements
| Method | CPU | RAM | GPU | Storage |
|--------|-----|-----|-----|---------|
| **Method 1** | Any | 4GB | None | Minimal |
| **Method 2** | Modern | 16GB | 8GB VRAM | ~3GB |
| **Method 3** | Modern | 16GB | 8GB VRAM | ~800MB |

---

# Key Takeaways

## Critical Success Factors
1. **Three complementary approaches** address different needs
2. **Statistical validation essential** - Cochran sampling required
3. **Immediate deployment possible** with Method 1
4. **Privacy-first option available** with Method 2
5. **Future scalability** through Method 3 training

## Bottom Line
**Start with Method 1, validate with Cochran sampling, evolve to Method 3**

**99.97% cost reduction vs manual processing**

---

# Questions & Discussion

## Contact & Resources
- **Full Documentation**: README.md with complete specs
- **Implementation Code**: Available in `/src/` and `/notebooks/`
- **Live Results**: JSON outputs in `/output/` directory

**Thank you for your attention!**
