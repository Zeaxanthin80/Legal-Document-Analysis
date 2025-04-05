from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import logging
import os # Import os module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    """A class to handle document search and retrieval operations, with persistence."""
    
    def __init__(self, index_path: str = "faiss_index", embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the SearchEngine, loading index if it exists.
        
        Args:
            index_path (str): Path to save/load the FAISS index.
            embedding_model (str): Name of the OpenAI embedding model to use.
        """
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        # self.documents = [] # We don't need to store all documents in memory if index is persisted
        # It's better to rely on the vector store for retrieval

        # Ensure index directory exists
        os.makedirs(self.index_path, exist_ok=True)

        # Try to load existing index
        self._load_index()
        
    def _load_index(self):
        """Load the FAISS index from the specified path if it exists."""
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                # FAISS requires allow_dangerous_deserialization=True for pickle loading
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_path}: {e}. Will create a new one.")
                self.vector_store = None
        else:
            logger.info(f"No existing FAISS index found at {self.index_path}. A new index will be created when documents are added.")
            self.vector_store = None

    def _save_index(self):
        """Save the FAISS index to the specified path."""
        if self.vector_store:
            try:
                logger.info(f"Saving FAISS index to {self.index_path}")
                self.vector_store.save_local(self.index_path)
                logger.info("FAISS index saved successfully.")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self.index_path}: {e}")
        else:
            logger.warning("No vector store to save.")
            
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the search engine index and save it.

        Args:
            documents (List[Document]): List of documents to add
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return
            
        if self.vector_store is None:
            logger.info(f"Creating new FAISS index with {len(documents)} documents.")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            logger.info(f"Adding {len(documents)} documents to existing FAISS index.")
            self.vector_store.add_documents(documents)
        
        # Save the index after adding documents
        self._save_index()

    def semantic_search(self, 
                       query: str, 
                       k: int = 5, 
                       filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform semantic search using the persisted index.
        """
        if self.vector_store is None:
            logger.warning("FAISS index is not loaded or created. Please load documents first.")
            return []
        
        logger.info(f"Performing semantic search for: '{query}' with k={k} and filter={filter}")
        try:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
            logger.info(f"Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    def keyword_search(self, 
                      query: str, 
                      k: int = 5,
                      filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        [Note: This implementation currently searches in-memory documents loaded 
        during *this* session, not the persisted index. Modify if persistence 
        across sessions is needed for keyword search.]
        """
        logger.warning("Keyword search currently operates on in-memory documents from this session only.")
        # We need to fetch relevant docs from the vector store first if we want persistence
        # This is a placeholder for a more robust implementation
        try:
            # Retrieve a larger number of docs potentially matching the filter
            # This is inefficient but demonstrates the concept
            candidate_docs = self.vector_store.similarity_search("", k=1000, filter=filter) if self.vector_store else []
        except Exception as e:
            logger.error(f"Error retrieving documents for keyword search: {e}")
            candidate_docs = []

        if not candidate_docs:
            logger.warning("No candidate documents found for keyword search (check filters or load docs).")
            return []

        results_with_scores = []
        query_terms = query.lower().split()
        
        logger.info(f"Performing keyword search on {len(candidate_docs)} candidates for: '{query}'")
        for doc in candidate_docs:                    
            # Calculate score
            score = sum(1 for term in query_terms if term in doc.page_content.lower())
            if score > 0:
                results_with_scores.append((doc, score))
                
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_results = [doc for doc, _ in results_with_scores[:k]]
        logger.info(f"Found {len(final_results)} keyword results.")
        return final_results

    def regex_search(self, 
                    pattern: str, 
                    k: int = 5,
                    filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        [Note: This implementation currently searches in-memory documents loaded 
        during *this* session, not the persisted index. Modify if persistence 
        across sessions is needed for regex search.]
        """
        logger.warning("Regex search currently operates on in-memory documents from this session only.")
        # Similar limitation as keyword_search regarding persistence
        try:
            candidate_docs = self.vector_store.similarity_search("", k=1000, filter=filter) if self.vector_store else []
        except Exception as e:
            logger.error(f"Error retrieving documents for regex search: {e}")
            candidate_docs = []

        if not candidate_docs:
            logger.warning("No candidate documents found for regex search (check filters or load docs).")
            return []
            
        results = []
        logger.info(f"Performing regex search on {len(candidate_docs)} candidates for pattern: '{pattern}'")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            count = 0
            for doc in candidate_docs:
                # Check regex
                if regex.search(doc.page_content):
                    results.append(doc)
                    count += 1
                    if count >= k:
                        break # Stop once we have k matches
                        
        except re.error as e:
            logger.error(f"Invalid regex pattern: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error during regex application: {e}")
            return []
            
        logger.info(f"Found {len(results)} regex results.")
        return results

    def hybrid_search(self,
                     query: str,
                     metadata_filters: Optional[Dict[str, Any]] = None,
                     k: int = 5) -> List[Document]:
        """
        Perform semantic search with metadata filtering.
        (This is now equivalent to calling semantic_search directly with a filter).

        Args:
            query (str): Search query
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters
            k (int): Number of results to return

        Returns:
            List[Document]: List of relevant documents
        """
        logger.info("Performing hybrid search (semantic + metadata filter).")
        # Directly call semantic_search which now handles filtering
        return self.semantic_search(query=query, k=k, filter=metadata_filters) 