# Legal Document Analysis with LangChain

This project provides a comprehensive solution for analyzing legal documents, specifically focused on Chapter 39, Florida Statutes, using LangChain and modern NLP techniques.

## Overview and Features

This project builds a **Legal Document Analysis Assistant** designed to make dense legal documents easier to understand and query. It combines document processing, vector search, AI-powered analysis, and user-friendly interfaces.

**Core Components:**

*   **Document Processing & Indexing:**
    *   Ingests PDF legal documents (initially Chapter 39, FL Statutes).
    *   Extracts text, splits it into manageable chunks, and generates vector embeddings.
    *   Stores embeddings in a FAISS vector index (`data/processed/faiss_index`) for efficient semantic search (finding relevant text based on meaning).

*   **Database Management (SQLite):**
    *   Uses `data/processed/metadata.db` to track processed documents and analysis results.
    *   `documents` table: Stores file paths, content hashes (to avoid reprocessing unchanged files), chunk counts, and indexing timestamps.
    *   `analysis_results` table: Stores outputs (e.g., summaries, topic analyses) as JSON, linked to specific documents and analysis types.

*   **AI-Powered Analysis:**
    *   Leverages a `LegalClassifier` (using Large Language Models like OpenAI's) to perform analyses (e.g., summarization) on document chunks.
    *   Stores generated analysis results in the database for later retrieval.

*   **Chatbot (Retrieval-Augmented Generation - RAG):**
    *   Provides a conversational interface for asking questions about the indexed documents.
    *   Retrieves relevant text chunks from the FAISS index based on the question's meaning.
    *   Sends the question and retrieved context to an LLM to generate answers grounded in the source document.

*   **User Interfaces:**
    *   **Command-Line Interface (CLI):** (`src.interface.cli`) For backend tasks: loading documents (`load`) and running analyses (`analyze <file> --type <type>`).
    *   **Streamlit Web Application:** (`src.interface.app`) Provides a user-friendly chat interface and allows viewing stored analysis results.

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