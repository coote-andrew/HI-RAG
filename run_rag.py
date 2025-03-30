from langchain_rag import LangChainRAG
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def print_usage():
    print("\nUsage:")
    print("  python run_rag.py                 - Interactive mode")
    print("  python run_rag.py index           - Index new files")
    print("  python run_rag.py reindex all     - Reindex all files")
    print("  python run_rag.py reindex <file>  - Reindex specific file")
    print("  python run_rag.py export <file>   - Export chunks from file to JSON")

def main():
    # Initialize the RAG system
    rag = LangChainRAG()
    
    # Check if documents directory exists
    if not Path("documents").exists():
        Path("documents").mkdir(exist_ok=True)
        print("Created 'documents' directory")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "export":
            if len(sys.argv) < 3:
                print("Error: Please specify a filename to export")
                print_usage()
                return
                
            filename = sys.argv[2]
            # Check if file exists in documents directory
            if not (Path("documents") / filename).exists():
                print(f"Error: File {filename} not found in documents directory")
                return
                
            print(f"Exporting chunks from {filename}...")
            success = rag.export_document_chunks(filename)
            if success:
                print(f"Successfully exported chunks to exports/")
            else:
                print("Export failed")
            return
        elif sys.argv[1] == "index":
            print("Indexing all files in documents folder...")
            success = rag.index_all_files()
            if success:
                print("Indexing completed successfully")
            else:
                print("Indexing failed")
            return
        elif sys.argv[1] == "reindex":
            if len(sys.argv) < 3:
                print("Error: Please specify 'all' or a filename to reindex")
                print_usage()
                return
                
            target = sys.argv[2]
            if target == "all":
                print("Reindexing all files...")
                # Delete all entries from ProcessedFile table
                rag.clear_processed_files()
                # Reindex everything
                success = rag.index_all_files()
                if success:
                    print("Reindexing completed successfully")
                else:
                    print("Reindexing failed")
            else:
                # Check if file exists
                filepath = Path("documents") / target
                if not filepath.exists():
                    print(f"Error: File {target} not found in documents directory")
                    return
                    
                print(f"Reindexing {target}...")
                # Remove from processed files table
                rag.remove_processed_file(str(filepath))
                # Reindex the file
                success = rag.load_file(str(filepath))
                if success:
                    print(f"Reindexed {target} successfully")
                else:
                    print(f"Failed to reindex {target}")
            return
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print_usage()
            return
    
    # Check for unprocessed files
    unprocessed = rag.get_unprocessed_files()
    if unprocessed:
        print("\nWARNING: The following files have not been indexed:")
        for file in unprocessed:
            print(f"- {file}")
        print("\nRun 'python run_rag.py index' to index these files")
    
    # Continue with interactive mode
    print("\nEntering interactive mode...")
    while True:
        command = input("\nEnter your command ('chunks' to view chunks, 'ask' for RAG response, "
                       "'direct' for LLM-only response, 'history' to view processing history, 'quit' to exit): ")
        
        if command.lower() == 'quit':
            break
        elif command.lower() == 'history':
            history = rag.get_processing_history()
            print("\nProcessing History:")
            for entry in history:
                print(f"\nFile: {entry['filepath']}")
                print(f"Processed: {entry['timestamp']}")
                print(f"Status: {entry['status']}")
                print(f"Chunks: {entry['num_chunks']} (size={entry['chunk_size']}, overlap={entry['chunk_overlap']})")
                if entry['error_message']:
                    print(f"Error: {entry['error_message']}")
        elif command.lower() == 'chunks':
            documents = rag.get_all_documents()
            print(f"\nFound {len(documents)} chunks:")
            for i, doc in enumerate(documents, 1):
                print(f"\n--- Chunk {i} ---")
                print(f"Content: {doc['content'][:500]}...")  # Show first 500 chars
                print(f"Metadata: {doc['metadata']}")
                if doc['metadata'].get('type') == 'sql':
                    print(f"SQL Type: {doc['metadata'].get('statement_type')}")
                    print(f"Tables: {doc['metadata'].get('tables')}")
        elif command.lower() == 'direct':
            question = input("Enter your question (LLM-only): ")
            response = rag.get_direct_response(question)
            print("\nDirect LLM Response:", response)
        elif command.lower() == 'ask':
            question = input("Enter your question (with RAG): ")
            
            # First show the retrieved documents
            print("\nRetrieved Documents:")
            docs = rag.get_relevant_documents(question)
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Retrieved Document {i} ---")
                print(f"Content: {doc['content'][:200]}...")
                print(f"Metadata: {doc['metadata']}")
                if doc['metadata'].get('type') == 'sql':
                    print(f"SQL Type: {doc['metadata'].get('statement_type')}")
                    print(f"Tables: {doc['metadata'].get('tables')}")
            
            # Then show the LLM response
            result = rag.get_response(question)
            print("\nRAG-enhanced Response:", result['response'])
            print("\nSources:", result['sources'])
        else:
            print("Unknown command. Use 'chunks', 'ask', 'direct', 'history', or 'quit'")

if __name__ == "__main__":
    main() 