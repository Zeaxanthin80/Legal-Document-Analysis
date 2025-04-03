import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from ..document_processing.loader import DocumentLoader
from ..search.engine import SearchEngine
from ..analysis.classifier import LegalClassifier

class LegalAnalysisCLI:
    """Command-line interface for legal document analysis."""
    
    def __init__(self):
        """Initialize the CLI components."""
        self.loader = DocumentLoader()
        self.search_engine = SearchEngine()
        self.classifier = LegalClassifier()
        
    def load_documents(self, directory: str) -> None:
        """Load documents from a directory."""
        print(f"Loading documents from {directory}...")
        documents = self.loader.load_documents(directory)
        self.search_engine.add_documents(documents)
        print(f"Loaded {len(documents)} documents.")
        
    def search_documents(self, query: str, search_type: str = "semantic") -> None:
        """Search documents using specified method."""
        if search_type == "semantic":
            results = self.search_engine.semantic_search(query)
        elif search_type == "keyword":
            results = self.search_engine.keyword_search(query)
        elif search_type == "regex":
            results = self.search_engine.regex_search(query)
        else:
            print(f"Unknown search type: {search_type}")
            return
            
        self._display_results(results)
        
    def analyze_documents(self, directory: str) -> None:
        """Analyze documents in a directory."""
        print(f"Analyzing documents in {directory}...")
        documents = self.loader.load_documents(directory)
        results = self.classifier.analyze_documents(documents)
        
        # Save results to file
        output_file = Path(directory) / "analysis_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Analysis results saved to {output_file}")
        
    def _display_results(self, results: List[Any]) -> None:
        """Display search results."""
        if not results:
            print("No results found.")
            return
            
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
            
def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Legal Document Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Load documents command
    load_parser = subparsers.add_parser("load", help="Load documents from a directory")
    load_parser.add_argument("directory", help="Directory containing documents")
    
    # Search documents command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--type", choices=["semantic", "keyword", "regex"],
                             default="semantic", help="Search type")
    
    # Analyze documents command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze documents")
    analyze_parser.add_argument("directory", help="Directory containing documents")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    cli = LegalAnalysisCLI()
    
    if args.command == "load":
        cli.load_documents(args.directory)
    elif args.command == "search":
        cli.search_documents(args.query, args.type)
    elif args.command == "analyze":
        cli.analyze_documents(args.directory)
        
if __name__ == "__main__":
    main() 