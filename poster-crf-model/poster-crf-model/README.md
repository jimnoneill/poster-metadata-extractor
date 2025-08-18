---
language: en
tags:
- scientific-text
- poster-extraction
- crf
- sequence-labeling
- ner
license: apache-2.0
metrics:
- f1
- precision
- recall
---

# Scientific Poster Metadata CRF Model

This model extracts structured metadata from scientific posters using a DistilBERT backbone with a CRF layer for sequence labeling.

## Model Description

- **Architecture**: DistilBERT + Linear + CRF
- **Task**: Named Entity Recognition for scientific posters
- **Labels**: Title, Authors, Affiliations, Methods, Results, Funding, Keywords
- **Base Model**: distilbert-base-uncased

## Label Scheme

The model uses BIO tagging:
- `B-TITLE`: Beginning of title
- `I-TITLE`: Inside title
- `B-AUTHOR`: Beginning of author name
- `I-AUTHOR`: Inside author name
- `B-AFFIL`: Beginning of affiliation
- `I-AFFIL`: Inside affiliation
- `B-METHOD`: Beginning of methods section
- `I-METHOD`: Inside methods section
- `B-RESULT`: Beginning of results
- `I-RESULT`: Inside results
- `B-FUND`: Beginning of funding information
- `I-FUND`: Inside funding information
- `B-KEYWORD`: Beginning of keyword
- `I-KEYWORD`: Inside keyword
- `O`: Outside any entity

## Usage

```python
from transformers import AutoTokenizer
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("jimnoneill/poster-metadata-crf")

# Load model (custom loading required for CRF layer)
# See the advanced_nlp_extraction notebook for full implementation
model = TransformerCRFModel.from_pretrained("jimnoneill/poster-metadata-crf")

# Process text
text = "Deep Learning for Medical Image Analysis"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs['predictions']
    
# Decode predictions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, label_id in zip(tokens, predictions[0]):
    label = ID_TO_LABEL[label_id]
    if label != 'O':
        print(f"{token}: {label}")
```

## Training Data

Trained on synthetic scientific poster data with manual annotations for demonstration purposes.

## Performance

- Training Loss: ~0.5
- Validation Loss: ~0.6
- Training on small synthetic dataset for demonstration

## Limitations

- English language only
- Trained on limited synthetic data
- Optimized for standard poster layouts
- May require fine-tuning for specific domains

## Citation

```bibtex
@misc{poster-metadata-crf,
  author = {Jim O'Neill},
  title = {Scientific Poster Metadata CRF},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/jimnoneill/poster-metadata-crf}
}
```
