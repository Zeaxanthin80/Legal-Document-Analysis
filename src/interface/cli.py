import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from ..document_processing.loader import DocumentLoader
from ..search.engine import SearchEngine
from ..analysis.classifier import LegalClassifier

INDEX_PATH = "faiss_index" # Define index path constant

class LegalAnalysisCLI:
    """Command-line interface for legal document analysis."""
    
    def __init__(self):
        """Initialize the CLI components, loading the index."""
        self.loader = DocumentLoader()
        # Pass the index path to the SearchEngine
        self.search_engine = SearchEngine(index_path=INDEX_PATH)
        self.classifier = LegalClassifier()
        
    def load_documents(self, directory: str) -> None:
        """Load documents from a directory."""
        print(f"Loading documents from {directory}...")
        documents = self.loader.load_documents(directory)
        self.search_engine.add_documents(documents)
        print(f"Loaded {len(documents)} documents.")
        
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
        
    def analyze_documents(self, directory: str, limit: int = 5) -> None:
        """Analyze documents in a directory (first few by default)."""
        print(f"Loading documents from {directory} for analysis...")
        documents = self.loader.load_documents(directory)
        
        if not documents:
            print("No documents found to analyze.")
            return

        # Limit the number of documents to analyze
        docs_to_analyze = documents[:limit]
        print(f"Analyzing the first {len(docs_to_analyze)} document chunks (out of {len(documents)} total)... This may take a while.")
        
        # Analyze the limited subset
        results = self.classifier.analyze_documents(docs_to_analyze)
        
        # Save results to file
        output_file = Path(directory) / "analysis_results.json"
        # Ensure parent directory exists (useful if directory is like data/raw)
        output_file.parent.mkdir(parents=True, exist_ok=True) 
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                # Ensure JSON can handle potential non-serializable data gracefully
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            print(f"Analysis results for the first {len(docs_to_analyze)} chunks saved to {output_file}")
        except Exception as e:
            print(f"Error saving analysis results to {output_file}: {e}")
        
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
    
    # Analyze documents command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze documents in a directory (loads them fresh, does not use index)")
    analyze_parser.add_argument("directory", help="Directory containing documents")
    # Add optional limit argument
    analyze_parser.add_argument("--limit", type=int, default=5, 
                              help="Limit the number of document chunks to analyze (default: 5)")
    
    args = parser.parse_args()
        
    cli = LegalAnalysisCLI()
    
    if args.command == "load":
        cli.load_documents(args.directory)
    elif args.command == "search":
        filters = parse_filters(args.filter or []) 
        cli.search_documents(args.query, args.type, filters)
    elif args.command == "analyze":
        # Pass the limit to the analyze_documents method
        cli.analyze_documents(args.directory, args.limit)
        
if __name__ == "__main__":
    main() 