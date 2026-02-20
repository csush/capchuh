# Capchuh

Measure the **generic-ness** of text to distinguish AI-generated content from human-generated content.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
python -m capchuh "Some text to analyze"
```

### Python API

```python
from capchuh import analyze

result = analyze("Some text to analyze")
print(result.raw_output)
```

## Run Tests

```bash
pytest
```
