from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    """A class to handle document search and retrieval operations."""
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the SearchEngine.
        
        Args:
            embedding_model (str): Name of the OpenAI embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self.documents = []
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the search engine.
        
        Args:
            documents (List[Document]): List of documents to add
        """
        self.documents.extend(documents)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of relevant documents
        """
        if self.vector_store is None:
            logger.warning("No documents have been added to the search engine")
            return []
            
        return self.vector_store.similarity_search(query, k=k)
    
    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform keyword-based search.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of relevant documents
        """
        results = []
        query_terms = query.lower().split()
        
        for doc in self.documents:
            score = sum(1 for term in query_terms if term in doc.page_content.lower())
            if score > 0:
                results.append((doc, score))
                
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:k]]
    
    def regex_search(self, pattern: str, k: int = 5) -> List[Document]:
        """
        Perform regex-based search.
        
        Args:
            pattern (str): Regular expression pattern
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of relevant documents
        """
        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            for doc in self.documents:
                if regex.search(doc.page_content):
                    results.append(doc)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {str(e)}")
            return []
            
        return results[:k]
    
    def metadata_filter(self, 
                       metadata_filters: Dict[str, Any],
                       k: int = 5) -> List[Document]:
        """
        Filter documents based on metadata.
        
        Args:
            metadata_filters (Dict[str, Any]): Dictionary of metadata filters
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of filtered documents
        """
        results = []
        for doc in self.documents:
            if all(doc.metadata.get(key) == value 
                  for key, value in metadata_filters.items()):
                results.append(doc)
                
        return results[:k]
    
    def hybrid_search(self,
                     query: str,
                     metadata_filters: Optional[Dict[str, Any]] = None,
                     k: int = 5) -> List[Document]:
        """
        Perform hybrid search combining semantic and metadata filtering.
        
        Args:
            query (str): Search query
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of relevant documents
        """
        # Get semantic search results
        semantic_results = self.semantic_search(query, k=k)
        
        # Apply metadata filters if provided
        if metadata_filters:
            filtered_results = []
            for doc in semantic_results:
                if all(doc.metadata.get(key) == value 
                      for key, value in metadata_filters.items()):
                    filtered_results.append(doc)
            return filtered_results[:k]
            
        return semantic_results 