# Legal Document Analysis with LangChain

This project provides a comprehensive solution for analyzing legal documents and Florida Statutes using LangChain and modern NLP techniques.

## Features

- Document Processing: Load and preprocess legal documents from various formats (PDF, TXT, HTML)
- Advanced Search: Implement full-text search, regex matching, and metadata filtering
- Legal Analysis: Classify documents, identify legal topics, and extract key entities
- Interactive Interface: CLI and chatbot interface for querying and summarizing statutes
- Optional Database Integration: Store and index processed documents

## Project Structure

```
.
├── src/                    # Source code
│   ├── document_processing/  # Document loading and preprocessing
│   ├── search/             # Search and retrieval functionality
│   ├── analysis/           # Document classification and analysis
│   └── interface/          # CLI and chatbot interfaces
├── data/                   # Data directory
│   ├── raw/               # Raw legal documents
│   └── processed/         # Processed documents
├── notebooks/             # Jupyter notebooks for analysis
├── configs/               # Configuration files
└── tests/                # Test cases
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Document Processing:
```python
from src.document_processing.loader import DocumentLoader
loader = DocumentLoader()
documents = loader.load_documents("data/raw/")
```

2. Search:
```python
from src.search.engine import SearchEngine
search = SearchEngine()
results = search.search("your query here")
```

3. Analysis:
```python
from src.analysis.classifier import LegalClassifier
classifier = LegalClassifier()
topics = classifier.classify_document(document)
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 