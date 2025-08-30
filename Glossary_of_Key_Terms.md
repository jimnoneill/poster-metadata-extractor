# Scientific Poster Metadata Extraction - Glossary of Key Terms

## A

**API (Application Programming Interface)**
- A set of protocols and tools for building software applications. In this project, used to access external language models like DeepSeek for text processing.

**Accuracy Estimate**
- Predicted performance metric (85-92% range) for each method. **Important**: All estimates are unvalidated and require Cochran sampling validation before production use.

**Affiliations**
- Institutional associations of authors (universities, research centers, companies) that must be extracted from scientific posters.

---

## B

**BIO Tagging**
- Named Entity Recognition annotation scheme using Beginning-Inside-Outside labels (e.g., B-AUTHOR, I-AUTHOR, O) to identify entity boundaries in text sequences.

**BioELECTRA**
- Biomedical domain-specific variant of ELECTRA transformer model, pre-trained on PubMed corpus. Ranks 2nd on BLURB biomedical NLP leaderboard.

**BLEU Score**
- Bilingual Evaluation Understudy score measuring similarity between machine-generated and reference text. Used for evaluating extracted methods/results accuracy (threshold: >0.7).

**BLURB Leaderboard**
- Biomedical Language Understanding and Reasoning Benchmark - standardized evaluation suite for biomedical NLP models.

**Bootstrapping Strategy**
- Using Methods 1 & 2 (DeepSeek/Qwen) to auto-label posters and generate training data for Method 3 (BioELECTRA+CRF), eliminating need for manual annotation.

---

## C

**Cochran Sampling**
- Statistical methodology for determining representative sample sizes for validation. Formula: n = (Z² × p × (1-p)) / e² with finite population correction.

**CRF (Conditional Random Fields)**
- Probabilistic graphical model used for structured prediction in sequence labeling tasks. Provides deterministic, hallucination-free entity extraction.

**CUDA**
- Compute Unified Device Architecture - parallel computing platform enabling GPU acceleration for machine learning models.

---

## D

**DeepSeek API**
- Cost-effective language model API service (~$0.003/poster) providing structured text extraction capabilities. 200x cheaper than GPT-4.

**Deterministic Extraction**
- Non-probabilistic, rule-based approach that produces consistent, reproducible results without hallucination risk (Method 3 characteristic).

---

## E

**Edit Distance**
- Levenshtein distance measuring character-level differences between strings. Used for fuzzy author name matching (threshold: <2).

**Encapsulation Efficiency (EE%)**
- Measure of drug loading success in nanoparticle formulations (from sample poster: PLGA 61.91% vs PLA/PEG 13.74%).

---

## F

**Few-Shot Prompting**
- Technique providing examples within prompts to guide model behavior without fine-tuning. Used in Method 2 for field-specific extraction.

**Finite Population Correction**
- Statistical adjustment applied to Cochran sample sizes when sampling from finite populations (formula: n_adjusted = n / (1 + (n-1)/N)).

---

## G

**GPU Memory Requirements**
- **Method 1**: None (API-based)
- **Method 2**: 8GB+ VRAM for quantized model
- **Method 3**: 8GB+ VRAM for training/inference

---

## H

**Hallucination**
- Generation of plausible but incorrect information by language models. Risk levels:
  - **Method 1**: Low-Medium (mitigated by structured prompts)
  - **Method 2**: Low (controlled generation)
  - **Method 3**: 0% (deterministic sequence labeling)

---

## J

**JSON Schema**
- Structured output format standardizing metadata fields across all methods:
```json
{
  "title": "string",
  "authors": [{"name": "string", "affiliations": ["string"], "email": "string"}],
  "summary": "string",
  "keywords": ["string"],
  "methods": "string",
  "results": "string",
  "references": [{"title": "string", "authors": "string", "year": "integer"}],
  "funding_sources": ["string"],
  "conference_info": {"location": "string", "date": "string"}
}
```

---

## L

**Language Model Quantization**
- Technique reducing model precision (32-bit → 8-bit) to decrease memory usage while maintaining performance. Used in Method 2 for efficiency.

---

## M

**Metadata Extraction**
- Systematic process of identifying and extracting structured information (title, authors, methods, results, etc.) from unstructured poster text.

**Microfluidics**
- Technology for precise manipulation of small fluid volumes (sample poster topic: nanoparticle synthesis using Passive Herringbone Mixer chips).

---

## N

**Named Entity Recognition (NER)**
- NLP task identifying and classifying entities (persons, organizations, methods, etc.) in text. Foundation of Method 3's approach.

**Nanoparticles (PLGA/PLA-PEG)**
- Drug delivery systems mentioned in sample poster:
  - **PLGA**: Poly(lactic-co-glycolic acid) - biodegradable polymer
  - **PLA-PEG**: Polylactic acid-polyethylene glycol copolymer

---

## O

**Overlap Coefficient**
- Similarity measure for keyword extraction validation (threshold: >0.6) calculated as |A ∩ B| / min(|A|, |B|).

---

## P

**PDF Text Extraction**
- Process using PyMuPDF (fitz) library to extract machine-readable text from poster PDF files for processing.

**PLGA Nanoparticles**
- Poly(lactic-co-glycolic acid) biodegradable polymer nanoparticles used for controlled drug delivery (sample poster focus).

**Processing Time**
- Method performance metrics:
  - **Method 1**: 5-15 seconds/poster
  - **Method 2**: 10-40 seconds single, ~1.1s batched (RTX 4090)
  - **Method 3**: <0.5 seconds/poster (estimated)

**Prompt Engineering**
- Craft of designing input prompts to optimize language model performance. Critical for Methods 1 & 2 accuracy.

**PyMuPDF (fitz)**
- Python library for PDF processing and text extraction. Used across all methods for initial text preprocessing.

---

## Q

**Qwen2.5-1.5B-Instruct**
- Small language model (1.5 billion parameters) optimized for instruction following. Used in Method 2 for local, privacy-preserving extraction.

**Quantization (8-bit)**
- Model compression technique reducing precision to fit larger models in limited GPU memory while maintaining performance.

---

## R

**RTX 4090 Batching**
- High-end GPU enabling efficient batch processing:
  - **Batch size**: 32 posters simultaneously
  - **Throughput**: 3,273 posters/hour, 26,182 posters/day

---

## S

**Semantic Similarity**
- Measure of meaning-level resemblance between texts (threshold: >0.8) used for title extraction validation.

**Sequence Labeling**
- NLP task assigning labels to each token in a sequence. Method 3 uses this for deterministic entity extraction.

**Structured Prompting**
- Technique using specific prompt formats and instructions to guide language models toward desired output structures.

---

## T

**Table 1 Metadata Fields**
- The eight required extraction targets:
  1. Title of the poster
  2. Authors (with affiliations)
  3. Summary of the poster
  4. Keywords
  5. Methods
  6. Results (main findings)
  7. References
  8. Funding source

**Temperature Parameter**
- Controls randomness in language model generation (0.1 used for consistency in Methods 1 & 2).

**Throughput Calculations**
- **Method 2 (RTX 4090 batched)**: 32 posters × 3.25 batches/minute × 60 minutes = 6,240 posters/hour theoretical maximum

---

## U

**Unvalidated Estimates**
- **Critical caveat**: All accuracy percentages (85-92%) are preliminary estimates requiring statistical validation through Cochran sampling before production use.

---

## V

**Validation Framework**
- Statistical methodology ensuring reliable accuracy assessment:
  - **Cochran sampling** for representative sample selection
  - **Field-specific metrics** for different metadata types
  - **95% confidence intervals** for statistical significance

---

## W

**Weighted F1-Score**
- Composite accuracy metric combining precision and recall across all metadata fields, weighted by field importance.

---

## Technical Architecture Terms

**BitsAndBytesConfig**
- Configuration object for model quantization settings in transformers library.

**CUDA Device Mapping**
- Automatic GPU memory allocation for distributed model loading across available hardware.

**Finite Population Correction Factor**
- Statistical adjustment: n_adjusted = n / (1 + (n-1)/N) where N is total population size.

**Herringbone Mixer Chip**
- Microfluidic device mentioned in sample poster for passive mixing in nanoparticle synthesis.

**Model Evaluation Metrics**
- **Exact Match**: Binary correctness measure
- **Edit Distance**: Character-level similarity (Levenshtein)
- **BLEU Score**: N-gram based similarity measure
- **Overlap Coefficient**: Set intersection similarity

---

## Cost Analysis Terms

**API Cost Structure**
- **DeepSeek**: $0.14 per 1M tokens (~$0.003 per poster)
- **GPT-4 Comparison**: 200x more expensive than DeepSeek
- **Manual Processing**: $5-10 per poster (human labor)
- **ROI**: 99.97% cost reduction with Method 1

**Electricity Costs**
- **Method 2**: Only ongoing cost for local processing (GPU power consumption)
- **Method 3**: Zero marginal cost after training completion

---

## Domain-Specific Scientific Terms

**Antimicrobial Resistance (AMR)**
- Medical challenge addressed in sample poster through improved drug delivery systems.

**Controlled Drug Delivery**
- Pharmaceutical approach for sustained, targeted medication release using nanoparticle carriers.

**Drug-Polymer Interactions**
- Chemical relationships between therapeutic compounds and carrier materials affecting release kinetics.

**Encapsulation Efficiency**
- Percentage of drug successfully loaded into nanoparticle carriers (key metric in sample poster).

---

## Implementation Status

**Production Ready**
- **Method 1**: Fully functional API integration
- **Method 2**: Complete local deployment with GPU optimization

**Demonstration Phase**
- **Method 3**: Framework and architecture defined, requires training data investment

**Validation Required**
- **All Methods**: Accuracy estimates require Cochran sampling validation before production deployment

---

## Hardware Requirements Summary

| Component | Method 1 | Method 2 | Method 3 |
|-----------|----------|----------|----------|
| **CPU** | Any modern | Modern multi-core | Modern multi-core |
| **RAM** | 4GB | 16GB+ | 16GB+ |
| **GPU** | Not required | 8GB+ VRAM | 8GB+ VRAM |
| **Storage** | Minimal | ~3GB (model) | ~800MB (trained) |
| **Network** | Internet required | Optional | Not required |

---

*This glossary covers all technical terms, domain concepts, and implementation details from the Scientific Poster Metadata Extraction project. Terms are organized alphabetically with cross-references and practical context for each method.*
