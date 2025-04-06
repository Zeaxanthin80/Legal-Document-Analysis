# --- Run this in your activated venv ---
import sys
# Add project root to path if needed, adjust '..' based on where you run this
# sys.path.insert(0, '..') 

from src.document_processing.loader import DocumentLoader
from pathlib import Path
import os

# Ensure OpenAI key is available if loader uses anything requiring it (currently doesn't)
# from dotenv import load_dotenv
# load_dotenv() 

pdf_path = Path("data/raw/Chapter 39 2022-10-24.pdf")
print(f"Checking PDF: {pdf_path.resolve()}")
print(f"File exists: {pdf_path.exists()}")

if pdf_path.exists():
    # Use default loader settings for consistency
    loader = DocumentLoader() 
    print("Loading and chunking document...")
    # Note: loader._preprocess_text cleans the text (lowercase etc.)
    document_chunks = loader.load_pdf(str(pdf_path))
    print(f"Generated {len(document_chunks)} chunks.")

    # Based on previous debugging, we know 39.501 starts in chunk 630
    target_chunk_index = 630
    start_phrase_lower = "39.501 petition for dependency.– (1)all proceedings seeking an adjudication" 
    
    target_body_phrase_lower = "the purpose of a petition seeking the adjudication of a child as a dependent child is the protection of the child and not the punishment of the person creating the condition of dependency"
    
    print("\n--- Examining Specific Chunks ---")
    
    # Check if our target chunk index is valid
    if target_chunk_index < len(document_chunks):
        print(f"\nCHUNK {target_chunk_index} FULL CONTENT:")
        chunk_content = document_chunks[target_chunk_index].page_content
        print("-" * 80)
        print(chunk_content)
        print("-" * 80)
        
        # Check if the start phrase is in this chunk
        if start_phrase_lower in chunk_content.lower():
            print(f"✓ Confirmed: Found start phrase in Chunk {target_chunk_index}")
        else:
            print(f"✗ WARNING: Start phrase NOT found in Chunk {target_chunk_index}")
        
        # Check if the target body phrase is in this chunk
        if target_body_phrase_lower in chunk_content.lower():
            print(f"✓ Found target body phrase in Chunk {target_chunk_index}")
        else:
            print(f"✗ Target body phrase NOT found in Chunk {target_chunk_index}")
        
        # Check the next chunk too
        if target_chunk_index + 1 < len(document_chunks):
            next_chunk_index = target_chunk_index + 1
            print(f"\nCHUNK {next_chunk_index} FULL CONTENT:")
            next_chunk_content = document_chunks[next_chunk_index].page_content
            print("-" * 80)
            print(next_chunk_content)
            print("-" * 80)
            
            # Check if the target body phrase is in the next chunk
            if target_body_phrase_lower in next_chunk_content.lower():
                print(f"✓ Found target body phrase in Chunk {next_chunk_index}")
            else:
                print(f"✗ Target body phrase NOT found in Chunk {next_chunk_index}")
        else:
            print("No next chunk available (reached end of document)")
    else:
        print(f"Error: Chunk {target_chunk_index} doesn't exist. Only {len(document_chunks)} chunks were generated.")
        
else:
    print("Error: PDF file not found.")
