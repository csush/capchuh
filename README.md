# Capchuh

Measure the **generic-ness** of text to distinguish AI-generated content from human-generated content.

AI outputs cluster tightly in embedding space (low variance, high similarity), while human outputs are more diverse. Capchuh embeds your text, computes cosine similarity to the centroid of all stored embeddings, and uses a MAD-based modified Z-score to flag outliers — outlier means likely human, non-outlier means likely AI-generic.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
git clone https://github.com/csush/capchuh.git
cd capchuh
uv sync
```

## Usage

### CLI

```bash
uv run python -m capchuh "Some text to analyze"
```

### Python API

```python
from capchuh import analyze

result = analyze("Some text to analyze")
print(result.raw_output)
# {"is_outlier": ..., "similarity": ..., "median_similarity": ..., "num_embeddings": ...}
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .
```
