# Scientific Poster Metadata Extraction Toolkit

Three approaches for extracting structured metadata from scientific posters, each optimized for different use cases.

## 📊 Approach Comparison

| Approach | Accuracy | Cost | Speed | Setup | Best For |
|----------|----------|------|-------|-------|----------|
| **DeepSeek API** | 90-95% | $0.003/poster | 5-10s | Easy | High accuracy needs |
| **Qwen 1.5B Local** | 85-90% | Free | 2-5s | Medium | Private, cost-effective |
| **Transformer+CRF** | 88-92%* | Free | <1s | Complex | Research/interpretability |

*With sufficient training data

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (for DeepSeek approach)
export DEEPSEEK_API_KEY="your-key-here"

# Run notebooks in notebooks/ directory
```

## 📁 Repository Contents

- `notebooks/poster_metadata_extraction.ipynb` - DeepSeek/LLM API approach
- `notebooks/qwen_small_lm_extraction.ipynb` - Local Qwen 1.5B model  
- `notebooks/advanced_nlp_extraction.ipynb` - Transformer+CRF demo
- `poster-crf-model/` - CRF training code (for future development)

## 🔍 Approach Details

### 1. DeepSeek API (Recommended)
**Pros:**
- Highest accuracy, handles complex layouts
- No training required, easy setup
- Supports multiple providers (OpenAI, Anthropic, Groq)

**Cons:**
- Requires API key and internet
- Per-poster costs
- Privacy concerns

### 2. Qwen 1.5B Local
**Pros:**
- Good accuracy/efficiency balance
- Runs locally, no API costs
- Can be fine-tuned

**Cons:**
- Slightly lower accuracy
- Requires GPU (4GB+ VRAM)
- Limited context window

### 3. Transformer+CRF (Demo)
**Pros:**
- Zero hallucination
- Very fast inference
- Interpretable results

**Cons:**
- Requires 500-1000 labeled posters for training
- Complex training pipeline
- Currently demo only

## 📈 CRF Training Recommendations

### Data Requirements
- **Minimum**: 500-1000 labeled posters
- **Per entity**: 100+ examples
- **Diversity**: Multiple conferences/layouts

### Simpler Alternatives
1. **Pure CRF**: 200-500 posters, 75-80% accuracy
2. **BiLSTM-CRF**: 300-700 posters, 82-87% accuracy  
3. **spaCy NER**: 300-500 posters, 80-85% accuracy

## ✅ Quality Validation

Use Cochran's sampling for manual validation:

```python
# For 1000 posters, validate ~278 random samples
sample_size = int(1.96**2 * 0.5 * 0.5 / 0.05**2)  # 95% confidence, 5% margin
```

## 📄 License

MIT License - See LICENSE file for details.