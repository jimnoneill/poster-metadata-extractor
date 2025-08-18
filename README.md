# Scientific Poster Metadata Extraction

## Executive Summary: A Three-Tiered Scientific Solution

The challenge of extracting structured metadata from scientific posters demands a methodologically rigorous approach that balances accuracy, cost, and scientific transparency. Our solution addresses this through a **progressive three-tiered architecture** designed to meet the original take-home task requirements while providing a clear pathway from rapid prototyping to production-ready scientific applications.

**Our recommended approach is Method 3: BioELECTRA+CRF** - a transformer-based conditional random field model specifically fine-tuned on biomedical literature. This approach eliminates hallucination risks entirely through deterministic sequence labeling, achieves the highest accuracy estimates (85-92%), and provides the fastest inference times (<0.5 seconds per poster). However, recognizing the practical constraints of training data requirements, we've developed **Methods 1 and 2 as both standalone solutions and data generation engines** for Method 3.

The elegance of our architecture lies in its **bootstrapping methodology**: Methods 1 (DeepSeek API) and 2 (Qwen2.5-1.5B local) serve dual purposes - they provide immediate extraction capabilities while simultaneously generating the 1,000+ labeled training examples needed to train our preferred BioELECTRA+CRF model. This creates a scientifically sound pathway from rapid deployment to long-term accuracy optimization, addressing the original task's requirement for both practical implementation and rigorous methodology.

Each method targets different operational contexts: Method 1 for immediate high-volume deployment, Method 2 for privacy-sensitive and resource-constrained environments, and Method 3 as our scientifically optimal endpoint that leverages insights from both predecessors.

## Important Note on Accuracy

**All accuracy estimates are unvalidated** - these are rough estimates based on limited testing and theoretical benchmarks. Actual accuracy can only be determined through proper validation using Cochran's random sampling methodology as outlined in this document. Please validate before production use.

## Original Task Requirements Addressed

This repository directly addresses the original take-home task requirements with our three-tiered approach:

### Pipeline Overview: Key Steps and Components

**Core Pipeline Steps:**
1. **PDF Text Extraction**: Use PyMuPDF (fitz) to extract text from poster PDFs
2. **Text Preprocessing**: Clean and structure extracted text for model consumption
3. **Metadata Extraction**: Apply selected method (API LLM, Local LLM, or Transformer+CRF)
4. **JSON Output**: Structure results according to Table 1 requirements (see below)
5. **Validation**: Apply Cochran sampling methodology for quality assessment

**Table 1 Metadata Fields (from original task):**
- **Title of the poster**
- **Authors (with affiliations)**  
- **Summary of the poster**
- **Keywords**
- **Methods**
- **Results (main findings)**
- **References**
- **Funding source**

**Key Components:**
- **Text Extraction Engine**: PyMuPDF for robust PDF processing
- **LLM Interface Layer**: Unified API for different model backends
- **Prompt Engineering Module**: Structured templates for consistent extraction
- **Validation Framework**: Statistical sampling and accuracy measurement
- **Output Standardization**: JSON schema compliance with required metadata fields

### Tools, Models, and Infrastructure

**Primary Tools (All Open Source):**
- **PyMuPDF**: PDF text extraction
- **Transformers**: HuggingFace model loading and inference
- **PyTorch**: Deep learning framework
- **NLTK/spaCy**: Natural language processing utilities
- **PyTorch-CRF**: Conditional Random Fields implementation
- **Jupyter**: Interactive development and demonstration

**Models by Method:**
- **Method 1**: DeepSeek-Chat (API-based, cost-effective alternative to GPT-4)
- **Method 2**: Qwen2.5-1.5B-Instruct (local inference, privacy-preserving)
- **Method 3**: BioELECTRA-base + CRF layer (domain-optimized, deterministic)

**Infrastructure:**
- **Development**: Local development with GPU support (RTX 4090 recommended)
- **Deployment Options**: CPU-only for Method 1, GPU-accelerated for Methods 2-3
- **API Integration**: OpenAI-compatible endpoints (DeepSeek, OpenAI, Anthropic, Groq)

### Assumptions and Dependencies

**Key Assumptions:**
- Poster PDFs contain extractable text (not image-only scans)
- Scientific posters follow standard academic formatting conventions
- Target metadata fields (Table 1) are present in poster content
- For Method 3: Training data can be generated via Methods 1-2 bootstrapping

**Dependencies:**
- **Python 3.8+**: Core runtime environment
- **CUDA-capable GPU**: Required for Methods 2-3 (8GB+ VRAM recommended)
- **API Keys**: DeepSeek API access for Method 1
- **Training Data**: 500-1000 labeled posters for Method 3 (generated via auto-labeling)
- **Validation Dataset**: Representative poster sample for Cochran sampling

### Pipeline Evaluation Framework

**Evaluation Approach:**
- **Cochran's Random Sampling**: Statistically significant validation methodology
- **Field-Specific Metrics**: Tailored accuracy measures per metadata type
- **Cross-Method Comparison**: Benchmarking across all three approaches
- **Statistical Validation**: 95% confidence intervals with finite population correction

**Specific Metrics:**
- **Title Extraction**: Exact match + semantic similarity (>0.8 threshold)
- **Author Detection**: Fuzzy string matching with edit distance <2
- **Keyword Extraction**: Overlap coefficient >0.6 with expert annotations  
- **Abstract Fields**: BLEU score >0.7 vs. expert summaries
- **Overall Accuracy**: Weighted F1-score across all metadata fields

**Sample Size Requirements (Cochran's Formula):**
- 1000 posters → 278 validation samples (27.8%)
- 10,000 posters → 370 validation samples (3.7%)
- 100,000+ posters → 383 validation samples (0.4%)

### Implementation Notes

**Current Implementation:**
- **Method 1**: Fully functional with enhanced structured prompting
- **Method 2**: Complete with 8-bit quantization and batching optimization
- **Method 3**: Demonstration framework only (requires training data)

**Differences from Ideal Pipeline:**
- **Method 3 Limitation**: Currently demo-only due to training data requirements
- **Hardware Constraints**: Optimized for single-GPU deployment vs. distributed inference
- **API Fallbacks**: Demo results provided when API keys unavailable

**Testing Instructions:**
1. Clone repository: `git clone https://github.com/jimnoneill/poster-metadata-extractor.git`
2. Install dependencies: `pip install -r requirements.txt`  
3. Configure API keys: `cp env.example .env` (edit as needed)
4. Run notebooks: Execute cells in `notebooks/01_method1_deepseek_api.ipynb`
5. Validate outputs: Check `output/` directory for generated JSON files

## Three-Method Approach

### Method 1: DeepSeek API Extraction
**Notebooks**: [`01_method1_deepseek_api.ipynb`](notebooks/01_method1_deepseek_api.ipynb)

Cost-effective API-based extraction using DeepSeek's language model.

**Performance Characteristics:**
- **Estimated Accuracy**: 85-90% (unvalidated - requires Cochran sampling validation)
- **Cost**: ~$0.003 per poster (200x cheaper than GPT-4)
- **Speed**: 5-15 seconds per poster  
- **Hallucination Risk**: Low-Medium (mitigated by structured prompts)
- **Setup**: Easy - just requires API key

**Best For**: Production systems with budget constraints, high-volume processing

### Method 2: Qwen Local Extraction
**Notebooks**: [`02_method2_qwen_local.ipynb`](notebooks/02_method2_qwen_local.ipynb)

Local small language model (1.5B parameters) for privacy-sensitive environments.

**Performance Characteristics:**
- **Estimated Accuracy**: 80-85% (unvalidated - requires Cochran sampling validation)
- **Cost**: $0 (runs locally, only electricity costs)
- **Speed**: 10-40 seconds per poster (single), ~1.1s per poster (RTX 4090 batched)
- **Hallucination Risk**: Low (structured prompting)
- **Setup**: Medium - requires model download and GPU memory

**RTX 4090 Batching Capacity:**
- **Recommended batch size**: 32 posters simultaneously
- **Throughput**: ~3,273 posters/hour, ~26,182 posters/day (8hrs)
- **Memory efficiency**: 8-bit quantization enables large-scale processing

**Best For**: Privacy-sensitive environments, budget-conscious deployments, edge computing

### Method 3: BioELECTRA+CRF (DEMO)
**Notebooks**: [`03_method3_bioelectra_crf_demo.ipynb`](notebooks/03_method3_bioelectra_crf_demo.ipynb)

**DEMONSTRATION ONLY** - Future possibility requiring 500-1000 labeled posters for training.

**Performance Characteristics (Estimated):**
- **Estimated Accuracy**: 85-92% (theoretical - based on BLURB benchmarks, requires training & validation)
- **Cost**: $0 (after training - local inference only)  
- **Speed**: <0.5 seconds per poster (fastest of all methods)
- **Hallucination Risk**: 0% (deterministic sequence labeling)
- **Setup**: Complex - requires extensive training data

**Training Requirements**: 500-1000 manually labeled poster PDFs with BIO annotations

**Auto-Labeling Plan**: The training data for Method 3 will be generated by auto-labeling 1,000 posters (or however many needed) using our top-performing Methods 1 (DeepSeek) and 2 (Qwen) to bootstrap the CRF training dataset. This approach leverages LLM-generated annotations as weak supervision for the final deterministic model.

## Approach Comparison

| Feature | Method 1 (DeepSeek) | Method 2 (Qwen Local) | Method 3 (BioELECTRA) |
|---------|--------------------|-----------------------|----------------------|
| **Accuracy** | 85-90% (unvalidated) | 80-85% (unvalidated) | 85-92% (theoretical) |
| **Cost per poster** | $0.003 | $0 | $0 |
| **Speed** | 5-15s | 10-40s | <0.5s |
| **Privacy** | External API | Local | Local |
| **Setup complexity** | Easy | Medium | Complex |
| **Hallucination** | Low-Med | Low | None |
| **Training required** | No | No | Yes (500-1000 posters) |

## Quality Validation Framework

### Cochran's Sampling for Manual Validation

We strongly recommend validating extraction quality using **Cochran's formula** for statistically significant sample size determination:

```
n = (Z² × p × (1-p)) / e²
n_adjusted = n / (1 + (n-1)/N)  # Finite population correction
```

**Where:**
- **Z** = 1.96 (95% confidence level)
- **p** = 0.5 (maximum variability assumption)
- **e** = 0.05 (±5% margin of error)
- **N** = total population size (number of posters)

**Sample Sizes by Dataset (with finite population correction):**
- **100 posters**: Validate ~80 randomly selected outputs (79.5%)
- **500 posters**: Validate ~217 randomly selected outputs (43.5%)
- **1000 posters**: Validate ~278 randomly selected outputs (27.8%)
- **10,000 posters**: Validate ~370 randomly selected outputs (3.7%)
- **100,000+ posters**: Validate ~383 randomly selected outputs (0.4%)

**Key Insight**: For smaller datasets (<1000), you must validate a high percentage. Only when scaling to tens of thousands of posters does the required validation percentage become practical (under 5%).

**Validation Process:**
1. Extract metadata from full dataset
2. Randomly sample using calculated sample size
3. Expert manual review of sampled outputs
4. Calculate accuracy metrics (precision, recall, F1)
5. Apply correction factors to full dataset if needed

This ensures statistically significant quality assessment across all methods.

### Accuracy Measurement Guidelines

**Field-specific accuracy calculation:**
- **Title**: Exact match or semantic similarity >0.8
- **Authors**: Name matching with fuzzy string matching (edit distance <2)
- **Keywords**: Overlap coefficient >0.6 with expert annotations
- **Methods/Results**: BLEU score >0.7 compared to expert summaries

## Project Structure

```
poster_project/
├── notebooks/
│   ├── 01_method1_deepseek_api.ipynb      # DeepSeek API extraction
│   ├── 02_method2_qwen_local.ipynb        # Qwen local model
│   └── 03_method3_bioelectra_crf_demo.ipynb # BioELECTRA demo
├── scripts/
│   ├── method1_deepseek_api.py            # API extraction script
│   ├── method2_qwen_local.py             # Local model script
│   └── method3_bioelectra_crf_demo.py    # Demo script
├── output/                               # Extraction results
├── src/                                 # Reusable modules
├── test-poster.pdf                      # Sample poster for testing
├── requirements.txt                     # Dependencies
└── README.md                           # This file
```

## Installation & Setup

### 1. Environment Setup
```bash
git clone https://github.com/jimnoneill/poster-metadata-extractor.git
cd poster-metadata-extractor
pip install -r requirements.txt
```

### 2. API Configuration (Method 1)
```bash
cp env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

### 3. GPU Setup (Method 2)
For optimal performance with Qwen local model:
- CUDA-capable GPU with 8GB+ VRAM
- PyTorch with CUDA support

## Usage

### Quick Start
```python
# Method 1: DeepSeek API
from scripts.method1_deepseek_api import extract_poster_metadata
results = extract_poster_metadata("your-poster.pdf")

# Method 2: Qwen Local  
from scripts.method2_qwen_local import extract_poster_metadata_qwen
results = extract_poster_metadata_qwen("your-poster.pdf")

# Method 3: Demo only
from scripts.method3_bioelectra_crf_demo import bioelectra_crf_demo
demo_results = bioelectra_crf_demo()
```

### Notebook Execution
All notebooks are ready to run with pre-executed outputs:
1. Open desired method notebook in Jupyter
2. Set API keys if using Method 1
3. Run all cells to see extraction results

## Key Technologies by Method

### Method 1 (DeepSeek API)
- **DeepSeek Chat**: Cost-effective LLM with competitive performance
- **OpenAI-compatible API**: Easy integration
- **Structured prompting**: JSON schema enforcement
- **PyMuPDF**: PDF text extraction

### Method 2 (Qwen Local)
- **Qwen2.5-1.5B-Instruct**: Compact multilingual model
- **Transformers**: HuggingFace model loading
- **BitsAndBytes**: 8-bit quantization for memory efficiency
- **Few-shot prompting**: Task-specific extraction

### Method 3 (BioELECTRA Demo)
- **BioELECTRA**: Biomedical domain-optimized transformer (2nd on BLURB)
- **Conditional Random Fields**: Sequence labeling for entity extraction
- **BIO tagging**: Named entity recognition scheme
- **PyTorch CRF**: Implementation of CRF layer

## Recommendations

### For Production Use
1. **Start with Method 1** (DeepSeek API) for immediate deployment
2. **Implement Cochran sampling** for quality validation  
3. **Consider Method 2** for privacy-sensitive applications
4. **Plan Method 3** as long-term solution with proper training data

### For BioELECTRA Training (Method 3)
- **Collect 500-1000 poster PDFs** with diverse layouts and fields
- **Manual BIO annotation** (~40-60 expert hours)  
- **Entity types**: Title, Authors, Affiliations, Methods, Results, Funding
- **Alternative simpler approaches**: Rule-based NER, spaCy custom models
- **Validation**: Cross-validation on held-out test set

## Process, Limitations, and Future Work

### Development Process

**Iterative Methodology:**
- **Phase 1**: Rapid prototyping with API-based solution (Method 1)
- **Phase 2**: Privacy-preserving local implementation (Method 2) 
- **Phase 3**: Scientific rigor through transformer+CRF architecture (Method 3)
- **Validation**: Cochran sampling framework for statistical significance

**Design Decisions:**
- **Multi-tiered approach** addresses different operational requirements
- **Bootstrapping strategy** leverages LLM capabilities for CRF training data generation
- **JSON output standardization** ensures consistency across all methods
- **Modular architecture** enables easy method comparison and selection

### Current Limitations

**Technical Constraints:**
- **Method 3 Training**: Requires substantial labeled dataset (500-1000 posters)
- **GPU Dependencies**: Methods 2-3 require CUDA-capable hardware for optimal performance
- **Text-Only Processing**: Cannot handle image-only or poorly scanned PDFs
- **Single-Language Support**: Optimized for English academic papers

**Validation Limitations:**
- **Accuracy Estimates**: Based on limited testing, require proper validation
- **Domain Specificity**: Tested primarily on biomedical/engineering posters
- **Scale Testing**: Not yet validated on large-scale deployments (>10K posters)

### Future Enhancements

**Immediate Improvements (3-6 months):**
- **Complete Method 3 Training**: Generate 1000+ labeled examples using Methods 1-2
- **OCR Integration**: Add image processing for scanned posters using Tesseract/PaddleOCR
- **Multilingual Support**: Extend to Spanish, French, German scientific literature
- **Batch Processing**: Implement distributed processing for large poster collections

**Advanced Developments (6-12 months):**
- **Multi-modal Architecture**: Incorporate visual layout analysis using LayoutLM
- **Domain Adaptation**: Fine-tune models for specific scientific disciplines
- **Active Learning**: Implement uncertainty-based sample selection for validation
- **Real-time API**: Deploy as microservice with REST API for integration

**Research Extensions (1+ years):**
- **Cross-lingual Transfer**: Leverage multilingual transformers for global poster analysis
- **Temporal Analysis**: Track research trend evolution across poster collections
- **Graph-based Extraction**: Model author-institution-topic relationships
- **Automated Quality Assessment**: Self-monitoring extraction confidence scoring

### Evaluation Recommendations

**Before Production Use:**
1. **Conduct Cochran Sampling**: Validate accuracy on representative poster sample
2. **Domain Testing**: Evaluate performance across different scientific fields  
3. **Scale Assessment**: Test throughput and accuracy on large poster collections
4. **User Studies**: Gather feedback from scientific librarians and researchers

**Success Metrics:**
- **Accuracy**: >90% field-level accuracy on validation set
- **Coverage**: Extract ≥7 of 8 Table 1 metadata fields per poster
- **Throughput**: Process ≥1000 posters/hour on standard hardware
- **User Satisfaction**: ≥85% user acceptance in library/repository contexts

## License
MIT License - see LICENSE file for details.

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## Citation
```bibtex
@software{oneill2024poster,
  title={Scientific Poster Metadata Extraction Toolkit},
  author={O'Neill, Jim},
  year={2024},
  url={https://github.com/jimnoneill/poster-metadata-extractor}
}
```