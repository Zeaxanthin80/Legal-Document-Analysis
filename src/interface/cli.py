import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging # Import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from ..document_processing.loader import DocumentLoader
from ..search.engine import SearchEngine
from ..analysis.classifier import LegalClassifier
from ..storage.database import DatabaseManager, calculate_file_hash # Import DB components

INDEX_PATH = "faiss_index" # Define index path constant

# Configure logging (if not already configured elsewhere)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LegalAnalysisCLI:
    """Command-line interface for legal document analysis."""
    
    def __init__(self):
        """Initialize the CLI components, loading the index."""
        self.loader = DocumentLoader()
        # Pass the index path to the SearchEngine
        self.search_engine = SearchEngine(index_path=INDEX_PATH)
        self.classifier = LegalClassifier()
        self.db_manager = DatabaseManager() # Initialize DatabaseManager
        
    def load_documents(self, directory: str) -> None:
        """Load documents from a directory, checking the database to avoid reprocessing."""
        logging.info(f"Starting document loading process for directory: {directory}")
        source_dir = Path(directory)
        if not source_dir.is_dir():
            logging.error(f"Source directory not found: {directory}")
            return

        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Iterate through potential document files (e.g., PDFs)
        # Adjust the glob pattern if other document types are supported
        for file_path in source_dir.glob('*.pdf'): 
            relative_path_str = str(file_path.relative_to(source_dir.parent)) # Store path relative to project/data root
            logging.info(f"Checking file: {relative_path_str}")

            try:
                file_hash = calculate_file_hash(file_path)
                if not file_hash:
                    logging.warning(f"Could not calculate hash for {relative_path_str}, skipping.")
                    error_count += 1
                    continue

                # Check if document is already processed and unchanged
                if self.db_manager.check_document_processed(relative_path_str, file_hash):
                    logging.info(f"Skipping already processed and unchanged file: {relative_path_str}")
                    skipped_count += 1
                    continue

                # If not skipped, process the document
                logging.info(f"Processing file: {relative_path_str}")
                # Use the specific load_pdf method from DocumentLoader
                document_chunks = self.loader.load_pdf(str(file_path))
                
                if not document_chunks:
                    logging.warning(f"No chunks generated for {relative_path_str}, skipping indexing.")
                    error_count += 1
                    continue
                
                num_chunks = len(document_chunks)
                logging.info(f"Generated {num_chunks} chunks for {relative_path_str}. Adding to index...")
                
                # Add chunks to the search engine (FAISS index)
                self.search_engine.add_documents(document_chunks)
                logging.info(f"Successfully added chunks from {relative_path_str} to the index.")

                # Add record to the database after successful indexing
                doc_id = self.db_manager.add_document(relative_path_str, file_hash, num_chunks)
                if doc_id is None:
                     logging.error(f"Failed to add document record to database for {relative_path_str}")
                     # Decide if you want to treat this as a fatal error or just log it
                     error_count += 1
                else:
                     logging.info(f"Successfully added document record (doc_id={doc_id}) for {relative_path_str}")
                     processed_count += 1

            except Exception as e:
                logging.error(f"Error processing file {relative_path_str}: {e}", exc_info=True)
                error_count += 1

        logging.info("Document loading process finished.")
        print(f"\nLoad Summary:")
        print(f"  - Successfully processed and indexed: {processed_count} files")
        print(f"  - Skipped (already processed, unchanged): {skipped_count} files")
        print(f"  - Errors encountered: {error_count} files")

    def search_documents(self, 
                         query: str, 
                         search_type: str = "semantic", 
                         filters: Optional[Dict[str, Any]] = None) -> None:
        """Search documents using specified method and optional filters."""
        print(f"Searching for '{query}' using {search_type} search...", end='')
        if filters:
            print(f" with filters: {filters}")
        else:
            print()
            
        if search_type == "semantic":
            results = self.search_engine.semantic_search(query, filter=filters)
        elif search_type == "keyword":
            results = self.search_engine.keyword_search(query, filter=filters)
        elif search_type == "regex":
            # Note: For regex, the query is the pattern
            results = self.search_engine.regex_search(query, filter=filters)
        else:
            print(f"Unknown search type: {search_type}")
            return
            
        self._display_results(results)
        
    def analyze_document(self, file_path_str: str, analysis_type: str) -> None:
        """Analyze a single document and store results in the database."""
        logging.info(f"Starting analysis for document: {file_path_str}, type: {analysis_type}")
        file_path = Path(file_path_str)

        if not file_path.is_file():
            logging.error(f"Document file not found: {file_path_str}")
            print(f"Error: File not found at '{file_path_str}'")
            return

        # Convert to relative path if needed
        try:
            # Handle paths that might start with 'data/'
            if file_path_str.startswith('data/'):
                relative_path_str = file_path_str[5:]  # Remove 'data/' prefix
            else:
                relative_path_str = file_path_str
            
            # Convert forward slashes to backslashes to match database storage
            relative_path_str = relative_path_str.replace('/', '\\')
            
            logging.info(f"Using relative path for DB lookup: {relative_path_str}")
        except ValueError:
            logging.error(f"Could not determine relative path for: {file_path_str}")
            print(f"Error: Invalid file path format: {file_path_str}")
            return

        # Get the doc_id from the database
        doc_id = self.db_manager.get_doc_id_by_path(relative_path_str)
        if doc_id is None:
            logging.error(f"Document '{relative_path_str}' not found in the database. Please load it first using the 'load' command.")
            print(f"Error: Document '{file_path_str}' has not been loaded/indexed yet.")
            return

        logging.info(f"Found document in database (doc_id={doc_id}). Loading chunks...")

        # 2. Load document chunks (using the original, potentially absolute path)
        try:
            # Assuming load_pdf handles loading a single file
            document_chunks = self.loader.load_pdf(str(file_path)) 
            if not document_chunks:
                logging.error(f"No chunks generated for document: {file_path_str}")
                print(f"Error: Could not load or chunk document '{file_path_str}'.")
                return
            logging.info(f"Loaded {len(document_chunks)} chunks for analysis.")
        except Exception as e:
            logging.error(f"Error loading/chunking document {file_path_str}: {e}", exc_info=True)
            print(f"Error loading document: {e}")
            return

        # 3. Perform the analysis (assuming classifier handles a list of chunks)
        # TODO: Adapt this based on how LegalClassifier actually works!
        #       Does it need a specific analysis type passed? Does it return different structures?
        logging.info(f"Performing '{analysis_type}' analysis on {len(document_chunks)} chunks... This may take time.")
        try:
            # We might need to adapt the classifier call based on the 'analysis_type'
            # For now, assume analyze_documents performs a default analysis
            analysis_result = self.classifier.analyze_documents(document_chunks) 
            if not analysis_result:
                 logging.warning(f"Analysis returned no result for doc_id={doc_id}, type='{analysis_type}'")
                 print("Analysis did not produce a result.")
                 return # Or handle differently?
            logging.info(f"Analysis complete for doc_id={doc_id}, type='{analysis_type}'")
        except Exception as e:
            logging.error(f"Error during analysis for doc_id={doc_id}, type='{analysis_type}': {e}", exc_info=True)
            print(f"Error during analysis: {e}")
            return

        # 4. Store the result in the database
        result_id = self.db_manager.add_analysis_result(doc_id, analysis_type, analysis_result)
        
        if result_id:
            print(f"Successfully stored '{analysis_type}' analysis result (result_id={result_id}) for document '{relative_path_str}' in the database.")
        else:
            print(f"Error: Failed to store analysis result for document '{relative_path_str}' in the database.")

    def _display_results(self, results: List[Any]) -> None:
        """Display search results."""
        if not results:
            print("\nNo results found.")
            return
            
        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            # Display all metadata
            metadata_str = ", ".join(f'{k}=\"{v}\"' for k, v in doc.metadata.items() if k != 'source')
            if metadata_str:
                print(f"Metadata: {metadata_str}")
            print(f"Content: {doc.page_content[:300]}...") # Show slightly more content

    # Ensure db_manager is closed properly, maybe add a close method or rely on __del__
    def __del__(self):
        """Ensure database connection is closed when CLI object is destroyed."""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()

def parse_filters(filter_args: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a list of 'key=value' strings into a dictionary."""
    if not filter_args:
        return None
    filters = {}
    for f in filter_args:
        if '=' not in f:
            print(f"Warning: Ignoring invalid filter format '{f}'. Use key=value.", file=sys.stderr)
            continue
        key, value = f.split('=', 1)
        filters[key.strip()] = value.strip()
    return filters if filters else None

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Legal Document Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)
    
    # Load documents command
    load_parser = subparsers.add_parser("load", help="Load documents from a directory and update the search index")
    load_parser.add_argument("directory", help="Directory containing documents")
    
    # Search documents command
    search_parser = subparsers.add_parser("search", help="Search the persisted document index")
    search_parser.add_argument("query", help="Search query (or regex pattern for type=regex)")
    search_parser.add_argument("--type", choices=["semantic", "keyword", "regex"],
                             default="semantic", help="Search type (default: semantic)")
    search_parser.add_argument("--filter", action='append', 
                             help="Metadata filter in key=value format (can be used multiple times)")
    
    # Analyze single document command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single document and store results in the database")
    analyze_parser.add_argument("file_path", help="Path to the document file (relative to project root or data dir)")
    analyze_parser.add_argument("--type", required=True, 
                              choices=["topic", "summary", "classification"], # Add more types as needed
                              help="Type of analysis to perform")
    # Remove the --limit argument as we process one file
    
    args = parser.parse_args()
        
    cli = LegalAnalysisCLI()
    
    if args.command == "load":
        cli.load_documents(args.directory)
    elif args.command == "search":
        filters = parse_filters(args.filter or []) 
        cli.search_documents(args.query, args.type, filters)
    elif args.command == "analyze":
        # Pass the file path and analysis type
        cli.analyze_document(args.file_path, args.type)
        
if __name__ == "__main__":
    main() 