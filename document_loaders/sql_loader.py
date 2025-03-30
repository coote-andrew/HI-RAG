from langchain.schema import Document
from pathlib import Path
import logging
import chardet
import re
from langchain_community.vectorstores.utils import filter_complex_metadata

logger = logging.getLogger(__name__)

class SQLLoader:
    """Simplified loader for SQL files that preserves comments and basic structure"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> list[Document]:
        """Load and parse SQL file into documents"""
        try:
            logger.info(f"Starting to load SQL file: {self.file_path}")
            
            # First detect the file encoding
            with open(self.file_path, 'rb') as f:
                raw_content = f.read()
            
            # Detect the encoding
            detected = chardet.detect(raw_content)
            encoding = detected['encoding']
            
            logger.info(f"Detected encoding {encoding} for file {self.file_path}")
            
            # Try multiple encodings if detection fails
            encodings_to_try = [
                encoding,  # Detected encoding
                'utf-8',
                'utf-8-sig',  # UTF-8 with BOM
                'cp1252',     # Windows default
                'iso-8859-1'  # Latin-1
            ]
            
            sql_content = None
            successful_encoding = None
            
            for enc in encodings_to_try:
                if not enc:
                    continue
                try:
                    sql_content = raw_content.decode(enc)
                    successful_encoding = enc
                    break
                except UnicodeDecodeError:
                    logger.debug(f"Failed to decode with {enc} encoding")
                    continue
            
            if sql_content is None:
                raise ValueError(f"Could not decode file with any of the attempted encodings: {encodings_to_try}")
            
            logger.info(f"Successfully loaded file using {successful_encoding} encoding")
            
            # Clean up the content
            sql_content = self._clean_sql_content(sql_content)
            logger.debug("Content cleaned")
            
            # Split into chunks based on GO statements or blank lines
            chunks = self._split_into_chunks(sql_content)
            logger.info(f"Split content into {len(chunks)} chunks")
            
            documents = []
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    logger.debug(f"Processing chunk {i+1}")
                    # Extract comments from the chunk
                    comments = self._extract_comments(chunk)
                    tables = self._extract_table_names(chunk)
                    
                    # Format content to highlight comments
                    if comments:
                        content = f"""Comments:
{comments}

SQL:
{chunk}"""
                    else:
                        content = chunk
                    
                    # Create metadata dictionary and filter it
                    metadata = {
                        'source': str(self.file_path),
                        'type': 'sql',
                        'has_comments': bool(comments),
                        'tables': ', '.join(tables) if tables else '',
                        'encoding': successful_encoding,
                        'chunk_number': i + 1
                    }
                    
                    # Create document with filtered metadata
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    logger.debug(f"Added document with {len(content)} characters")
            
            logger.info(f"Successfully created {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading SQL file {self.file_path}: {str(e)}")
            logger.exception("Full traceback:")
            raise
    
    def _clean_sql_content(self, content: str) -> str:
        """Clean up SQL content by replacing problematic characters"""
        replacements = {
            '"': '"',    # Smart quotes
            '"': '"',
            ''': "'",    # Smart apostrophes
            ''': "'",
            '–': '-',    # En dash
            '—': '-',    # Em dash
            '…': '...',  # Ellipsis
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        return content
    
    def _split_into_chunks(self, content: str) -> list[str]:
        """Split SQL content into logical chunks"""
        # Split on GO statements or multiple blank lines
        chunks = re.split(r'\bGO\b|(\n\s*){3,}', content)
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    
    def _extract_comments(self, chunk: str) -> str:
        """Extract both inline and block comments from a chunk"""
        comments = []
        
        # Extract block comments
        block_comments = re.findall(r'/\*(.*?)\*/', chunk, re.DOTALL)
        comments.extend([comment.strip() for comment in block_comments])
        
        # Extract line comments
        for line in chunk.split('\n'):
            line = line.strip()
            if line.startswith('--'):
                comments.append(line[2:].strip())
        
        return '\n'.join(comments)
    
    def _extract_table_names(self, chunk: str) -> list[str]:
        """Extract potential table names using regex"""
        # Look for words after FROM, JOIN, INTO, UPDATE
        table_pattern = r'\b(FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        matches = re.findall(table_pattern, chunk, re.IGNORECASE)
        tables = [match[1] for match in matches]
        return list(set(tables))  # Remove duplicates 