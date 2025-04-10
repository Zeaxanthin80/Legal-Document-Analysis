import pytest
from pathlib import Path
from src.document_processing.loader import DocumentLoader
from src.search.engine import SearchEngine
from src.analysis.classifier import LegalClassifier
from langchain.schema import Document

@pytest.fixture
def sample_raw_text():
    """Provide sample raw text for preprocessing test."""
    return """
    Florida Statute 768.28 - Sovereign Immunity

    The state and its agencies and subdivisions shall be liable 
    for tort claims in the same manner and to the same extent as a private individual under like circumstances.
    
    """

@pytest.fixture
def expected_processed_text():
    """Provide expected text after preprocessing."""
    return "florida statute 768.28 - sovereign immunity the state and its agencies and subdivisions shall be liable for tort claims in the same manner and to the same extent as a private individual under like circumstances."

@pytest.fixture
def sample_document(expected_processed_text):
    """Create a sample document with preprocessed content."""
    # Note: This fixture now uses the expected processed text.
    # For tests involving loading, we'd need to mock file reads.
    return Document(
        page_content=expected_processed_text,
        metadata={"source": "test.txt", "type": "txt"}
    )

@pytest.fixture
def sample_document_pdf():
    """Create a second sample document with different metadata.""" 
    # Using slightly different content for variety
    content = "florida statute chapter 39 - proceedings relating to children this chapter shall be known as the unified juvenile justice act."
    return Document(
        page_content=content,
        metadata={"source": "test.pdf", "type": "pdf", "chapter": 39}
    )

@pytest.fixture
def search_engine_with_docs(sample_document, sample_document_pdf):
    """Create a SearchEngine instance and add both sample documents."""
    engine = SearchEngine()
    engine.add_documents([sample_document, sample_document_pdf])
    return engine

@pytest.fixture
def document_loader():
    """Create a DocumentLoader instance."""
    return DocumentLoader()

@pytest.fixture
def search_engine():
    """Create a SearchEngine instance."""
    return SearchEngine()

@pytest.fixture
def legal_classifier():
    """Create a LegalClassifier instance."""
    return LegalClassifier()

def test_document_loader_initialization(document_loader):
    """Test DocumentLoader initialization."""
    assert document_loader is not None
    assert document_loader.text_splitter is not None

def test_document_loader_preprocessing(document_loader, sample_raw_text, expected_processed_text):
    """Test the _preprocess_text method directly."""
    processed = document_loader._preprocess_text(sample_raw_text)
    assert processed == expected_processed_text

def test_search_engine_initialization(search_engine):
    """Test SearchEngine initialization."""
    assert search_engine is not None
    assert search_engine.embeddings is not None
    assert search_engine.vector_store is None
    assert search_engine.documents == []

def test_legal_classifier_initialization(legal_classifier):
    """Test LegalClassifier initialization."""
    assert legal_classifier is not None
    assert legal_classifier.llm is not None
    assert legal_classifier.nlp is not None

def test_search_engine_add_documents(search_engine, sample_document):
    """Test adding documents to search engine."""
    search_engine.add_documents([sample_document])
    assert len(search_engine.documents) == 1
    assert search_engine.vector_store is not None

def test_search_engine_semantic_search(search_engine, sample_document):
    """Test semantic search functionality with preprocessed content."""
    search_engine.add_documents([sample_document])
    results = search_engine.semantic_search("sovereign immunity") # Query is lowercase
    assert len(results) > 0
    # Check against the preprocessed content which is already lowercase
    assert "sovereign immunity" in results[0].page_content

def test_search_engine_keyword_search(search_engine, sample_document):
    """Test keyword search functionality with preprocessed content."""
    search_engine.add_documents([sample_document])
    # Keyword search in SearchEngine also converts query and content to lower
    results = search_engine.keyword_search("Florida Statute") 
    assert len(results) > 0
    # Check if the original concept (case-insensitive) is in the preprocessed text
    assert "florida statute" in results[0].page_content

def test_legal_classifier_classify_document(legal_classifier, sample_document):
    """Test document classification."""
    result = legal_classifier.classify_document(sample_document)
    assert result is not None
    assert "classification" in result
    assert "entities" in result
    assert "entity_counts" in result

def test_legal_classifier_extract_entities(legal_classifier, sample_document):
    """Test entity extraction."""
    entities = legal_classifier.extract_legal_entities(sample_document)
    assert isinstance(entities, list)
    assert all(isinstance(entity, dict) for entity in entities)
    assert all("text" in entity and "label" in entity for entity in entities)

def test_legal_classifier_analyze_legal_topics(legal_classifier, sample_document):
    """Test legal topic analysis."""
    topics = legal_classifier.analyze_legal_topics(sample_document)
    assert isinstance(topics, dict)

def test_legal_classifier_summarize_document(legal_classifier, sample_document):
    """Test document summarization."""
    summary = legal_classifier.summarize_document(sample_document)
    assert isinstance(summary, str)
    assert len(summary) > 0

# --- Tests for Metadata Filtering --- 

def test_search_engine_semantic_search_with_filter(search_engine_with_docs):
    """Test semantic search with metadata filtering."""
    # Filter for only PDF documents
    pdf_filter = {"type": "pdf"}
    results = search_engine_with_docs.semantic_search("juvenile justice", filter=pdf_filter)
    assert len(results) == 1, "Should only find the PDF document"
    assert results[0].metadata["type"] == "pdf"
    assert "juvenile justice" in results[0].page_content

    # Filter for only TXT documents
    txt_filter = {"type": "txt"}
    results = search_engine_with_docs.semantic_search("sovereign immunity", filter=txt_filter)
    assert len(results) == 1, "Should only find the TXT document"
    assert results[0].metadata["type"] == "txt"
    assert "sovereign immunity" in results[0].page_content

    # Filter for non-existent metadata
    non_existent_filter = {"category": "case_law"}
    results = search_engine_with_docs.semantic_search("tort claims", filter=non_existent_filter)
    assert len(results) == 0, "Should find no documents with this metadata"

def test_search_engine_keyword_search_with_filter(search_engine_with_docs):
    """Test keyword search with metadata filtering."""
    # Filter for only PDF documents
    pdf_filter = {"type": "pdf"}
    results = search_engine_with_docs.keyword_search("chapter 39", filter=pdf_filter)
    assert len(results) == 1, "Should only find the PDF document"
    assert results[0].metadata["type"] == "pdf"

    # Filter for only TXT documents
    txt_filter = {"type": "txt"}
    results = search_engine_with_docs.keyword_search("agencies", filter=txt_filter)
    assert len(results) == 1, "Should only find the TXT document"
    assert results[0].metadata["type"] == "txt"

    # Filter that excludes all matching content
    pdf_filter_wrong_content = {"type": "pdf"}
    results = search_engine_with_docs.keyword_search("sovereign immunity", filter=pdf_filter_wrong_content)
    assert len(results) == 0, "Should find no PDF containing 'sovereign immunity'"

def test_search_engine_regex_search_with_filter(search_engine_with_docs):
    """Test regex search with metadata filtering."""
    # Filter for only PDF documents, find chapter number
    pdf_filter = {"type": "pdf"}
    results = search_engine_with_docs.regex_search(r"chapter\s+\d+", filter=pdf_filter)
    assert len(results) == 1
    assert results[0].metadata["type"] == "pdf"

    # Filter for only TXT documents, find statute number
    txt_filter = {"type": "txt"}
    results = search_engine_with_docs.regex_search(r"statute\s+\d+\.\d+", filter=txt_filter)
    assert len(results) == 1
    assert results[0].metadata["type"] == "txt"

    # Filter that finds nothing
    no_match_filter = {"chapter": 40}
    results = search_engine_with_docs.regex_search(r"statute", filter=no_match_filter)
    assert len(results) == 0 