from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Optional, List, Union
from log_management.query_log import Session, QueryLog, ProcessedFile
from datetime import datetime
import json
import uuid
from document_loaders.sql_loader import SQLLoader
from langchain.schema import Document
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from sqlfluff import lint
import sqlparse

# Initialize logger for this module
logger = logging.getLogger(__name__)

class HealthIntelligenceResponse(BaseModel):
    chat_response: str              # The textual response from the LLM
    final_query: str                # The refined/processed query
    sql_query: str                  # SQL query generated (if applicable)
    suggested_refinements: Optional[List[str]] = None  # Suggested ways to improve the query

class LangChainRAG:
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None):
        # Initialize core components
        self.use_openai = use_openai
        
        # Set up the embedding model - this transforms text into vectors
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store with persistent storage
        self.vector_store = Chroma(
            persist_directory="data/chroma",
            embedding_function=self.embedding_model
        )
        
        # Set up the LLM based on configuration (OpenAI or Ollama)
        if use_openai:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when use_openai is True")
            self.openai_client = OpenAI(api_key=openai_api_key)
            # Create ChatOpenAI instance for LangChain compatibility
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                openai_api_key=openai_api_key,
                temperature=0  # Use deterministic outputs
            )
        else:
            # Use locally hosted Ollama model
            self.llm = Ollama(
                model="qwen2.5:7b",
                num_ctx=10384,  # Large context window
                num_gpu=1       # Use GPU acceleration
            )
        
        # Initialize QA chain with basic prompt
        prompt = PromptTemplate(
            template="""You are providing a helpful guide to writing a good Health Intelligence Query. You are helping a user ask a question about their data that can then be provided to our health intelligence team for them to analyse and provide an answer.
            The team has access to a range of data sources, including clinical, operational and administrative data - you will not see this.
            They can ask for data about the Royal Melbourne Hospital, including all data available.
            We use an EMR called Epic, and a PACS called iPM, we store admission details, notes, pathology, radiology, medications, as well as discrete documentation data, like observations or some examination/finding results - mainly those from allied health.
            Assume that all questions are about the Royal Melbourne Hospital, unless otherwise specified.
            Your job is to make sure the question is clear and concise and that the user has provided enough information for the health intelligence team to answer it.
            Use the following context to answer the question.
            Do not add time periods to the question, unless the user has specified a time period.
            Your job is to ask questions to the user to help them refine their query.
            If you don't know the answer, just say that you don't know.
            VERY VERY VERY IMPORTANT: ALWAYS USE CLARITY DATABASE For EPIC Queriest. NEVER USE CABOODLE.
            
            Context: {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"]
        )
        
        # Initialize retriever to get relevant documents from vector store
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={"k": 4}     # Return top 4 results
        )
        
        # Initialize QA chain that connects the LLM with the retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",         # "Stuff" means feed all context in one prompt
            retriever=self.retriever,
            return_source_documents=True # Include source documents in response
        )

        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)

        # Add SQL file handling
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.sql': SQLLoader
        }

        # Initialize SQL formatter settings
        self.sql_format_kwargs = {
            'reindent': True,
            'keyword_case': 'upper',
            'identifier_case': 'lower',
            'strip_comments': False,
            'wrap_after': 80
        }

    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        """Add documents to the vector store"""
        try:
            self.vector_store.add_texts(texts, metadatas=metadatas)
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def get_llm_response(self, prompt: str, format: Optional[dict] = None) -> Union[str, dict]:
        """Helper method to get response from either Ollama or OpenAI"""
        if self.use_openai:
            messages = [{"role": "user", "content": prompt}]
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"} if format else None
            )
            return response.choices[0].message.content
        else:
            return self.llm.invoke(prompt, format=format)

    def format_sql(self, sql_query: str) -> tuple[str, list[str]]:
        """
        Format and lint SQL query
        
        Args:
            sql_query: Raw SQL query string
            
        Returns:
            tuple: (formatted_query, list_of_lint_errors)
        """
        try:
            # Format the SQL query for readability
            formatted_sql = sqlparse.format(
                sql_query,
                **self.sql_format_kwargs
            )
            
            # Lint the SQL query to check for errors
            lint_results = lint(formatted_sql)
            lint_errors = [str(error) for error in lint_results]
            
            return formatted_sql, lint_errors
        except Exception as e:
            logger.error(f"Error formatting SQL: {str(e)}")
            return sql_query, [f"Formatting error: {str(e)}"]

    def get_response(self, question: str, conversation_id: Optional[str] = None, 
                    previous_query: Optional[str] = None, previous_sql: Optional[str] = None,
                    previous_chat: Optional[str] = None):
        """
        Generates a response to a health intelligence query based on relevant documents
        
        Parameters:
        - question: The user's query about health data
        - conversation_id: Optional ID to maintain conversation context
        - previous_query: Previously processed query for context
        - previous_sql: Previously generated SQL for context
        - previous_chat: Previous conversation for context
        
        Returns a structured response with guidance for formulating a good health query
        
        Args:
            question: The user's question
            conversation_id: Optional ID to track conversation history
            previous_query: Optional previous query text for context
            previous_sql: Optional previous SQL query for context
            previous_chat: Optional previous chat response for context
        """
        try:
            # Get more relevant documents
            docs = self.vector_store.similarity_search(question, k=10)  # Increased from 4 to 10
            
            # Build context including previous interactions if available
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Add previous context if available
            if any([previous_query, previous_sql, previous_chat]):
                previous_context = []
                if previous_query:
                    previous_context.append(f"Previous Query: {previous_query}")
                if previous_sql:
                    previous_context.append(f"Previous SQL: {previous_sql}")
                if previous_chat:
                    previous_context.append(f"Previous Chat: {previous_chat}")
                context = f"{' '.join(previous_context)}\n\n{context}"
            
            prompt = f"""You are providing guidance for Health Intelligence Queries.
            You are providing a helpful guide to writing a good Health Intelligence Query. You are helping a user ask a question about their data that can then be provided to our health intelligence team for them to analyse and provide an answer.
            The team has access to a range of data sources, including clinical, operational and administrative data - you will not see this.
            They can ask for data about the Royal Melbourne Hospital, including all data available.
            We use an EMR called Epic, and a PACS called iPM, we store admission details, notes, pathology, radiology, medications, as well as discrete documentation data, like observations or some examination/finding results - mainly those from allied health.
            Assume that all questions are about the Royal Melbourne Hospital, unless otherwise specified.
            Your job is to make sure the question is clear and concise and that the user has provided enough information for the health intelligence team to answer it.
            Use the following context to answer the question.
            Do not add time periods to the question, unless the user has specified a time period.
            Your job is to ask questions to the user to help them refine their query for up to 3 times from the start - the write a finalised health intelligence query.
            If you don't know the answer, just say that you don't know.
            The Royal Melbourne Hospital differes from the Epic base guide because it stores the "medical unit" on PAT_ENC_HSP.HOSP_SERV_C, which is linked to ZC_PAT_SERVICE.HOSP_SERV_C, with ZC_PAT_SERVICE.NAME as the unit name, rather than DEPARTMENT which you might expect.

            MEDICATION administration is stored in MAR_ADMIN_INFO, and will have an associated medication type, that associated medication type may have therapeutic classes or be in a "Grouper"
            VERY VERY VERY IMPORTANT: ALWAYS USE CLARITY DATABASE For EPIC Queriest. NEVER USE CABOODLE.
            
            Context: 
        
                    MEDICATION administration is stored in MAR_ADMIN_INFO, and will have an associated medication type, that associated medication type may have therapeutic classes or be in a "Grouper"
            VERY VERY VERY IMPORTANT: ALWAYS USE CLARITY DATABASE For EPIC Queriest. NEVER USE CABOODLE.
            
            Context: {context}
            User Question: {question}
            Previous Query (if any): {previous_query}

            Please provide a response in the following JSON format:
            {{
                "chat_response": "Your helpful response and questions to refine the query",
                "final_query": "The clear, human-readable query for the Health Intelligence team",
                "sql_query": "A draft SQL query that could be used to answer the question",
                "suggested_refinements": ["optional list", "of suggested", "query refinements"]
            }}
            """
            # Get structured response using selected model
            result = self.get_llm_response(
                prompt,
                format=HealthIntelligenceResponse.model_json_schema()
            )
            
            # Parse the response using Pydantic
            parsed_response = HealthIntelligenceResponse.model_validate_json(result)
            
            # Format and lint SQL query
            formatted_sql, lint_errors = self.format_sql(parsed_response.sql_query)
            parsed_response.sql_query = formatted_sql
            
            # Add lint errors to response if any
            if lint_errors:
                if parsed_response.suggested_refinements is None:
                    parsed_response.suggested_refinements = []
                parsed_response.suggested_refinements.extend([
                    f"SQL Lint: {error}" for error in lint_errors
                ])
            
            # Generate suggested refinements if there's a previous query
            suggested_refinements = []
            if previous_query:
                refinement_prompt = f"""Based on the previous query and the new context, suggest up to 3 refined versions of this query:
                Previous Query: {previous_query}
                User Question: {question}
                Context: {context[:500]}...
                
                Return as JSON array of strings."""
                
                refinements = self.get_llm_response(
                    refinement_prompt,
                    format={
                        "type": "array",
                        "items": {"type": "string"}
                    }
                )
                suggested_refinements = refinements if isinstance(refinements, list) else []
            
            return {
                'response': parsed_response.chat_response,
                'query_text': parsed_response.final_query,
                'sql_query': parsed_response.sql_query,
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'suggested_refinements': suggested_refinements
            }
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise



    def mark_file_processed(self, filepath: str, chunk_size: int, 
                          chunk_overlap: int, num_chunks: int, 
                          status: str = 'success', error_message: str = None):
        """Record a processed file in the database"""
        try:
            session = Session()
            processed_file = ProcessedFile(
                filepath=str(filepath),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                num_chunks=num_chunks,
                status=status,
                error_message=error_message
            )
            session.add(processed_file)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error marking file as processed: {str(e)}")

    def get_unprocessed_files(self, documents_dir: str = "documents") -> list[str]:
        """Check for unprocessed files in the documents directory"""
        try:
            # Get all supported files in the directory
            supported_files = []
            for ext in self.supported_extensions:
                supported_files.extend(
                    [str(p) for p in Path(documents_dir).glob(f"*{ext}")]
                )
            
            # Get list of successfully processed files from database
            session = Session()
            processed_files = {
                row.filepath for row in 
                session.query(ProcessedFile.filepath)
                .filter(ProcessedFile.status == 'success')
                .all()
            }
            session.close()
            
            # Return unprocessed files
            unprocessed = [
                f for f in supported_files 
                if f not in processed_files
            ]
            return unprocessed
        except Exception as e:
            logger.error(f"Error checking for unprocessed files: {str(e)}")
            return []

    def load_file(self, file_path: str, 
                  chunk_size: int = 1000,
                  chunk_overlap: int = 200,
                  metadata: Optional[dict] = None) -> bool:
        """Load a file, split it into chunks, and add it to the vector store"""
        try:
            file_path = Path(file_path)
            logger.info(f"Starting to load file: {file_path}")
            
            # Get appropriate loader based on file extension
            loader_class = self.supported_extensions.get(file_path.suffix.lower())
            if not loader_class:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            logger.info(f"Using loader class: {loader_class.__name__}")
            
            # Load documents using the appropriate loader
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            logger.info(f"Loaded {len(documents) if isinstance(documents, list) else 'unknown'} documents")
            logger.debug(f"First document type: {type(documents[0] if isinstance(documents, list) and documents else documents)}")
            
            # Ensure we have a list of Documents
            if not isinstance(documents, list):
                logger.warning(f"Documents is not a list, type: {type(documents)}")
                documents = [documents]
            
            # Convert strings to Documents if needed
            processed_documents = []
            for i, doc in enumerate(documents):
                logger.debug(f"Processing document {i}, type: {type(doc)}")
                if isinstance(doc, str):
                    logger.debug(f"Converting string to Document object")
                    processed_documents.append(Document(page_content=doc))
                elif isinstance(doc, Document):
                    logger.debug(f"Already a Document object")
                    processed_documents.append(doc)
                else:
                    logger.warning(f"Unknown document type: {type(doc)}")
                    processed_documents.append(Document(page_content=str(doc)))
            
            logger.info(f"Processed {len(processed_documents)} documents")
            
            # Create text splitter for chunking documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            
            # Split documents into manageable chunks
            logger.info("Splitting documents...")
            splits = text_splitter.split_documents(processed_documents)
            logger.info(f"Created {len(splits)} splits")
            
            # Add metadata if provided
            if metadata is None:
                metadata = {}
            metadata['source'] = str(file_path)
            
            logger.info("Adding metadata to splits...")
            for split in splits:
                if not hasattr(split, 'metadata'):
                    split.metadata = {}
                split.metadata.update(metadata)
            
            # Add chunks to vector store
            logger.info("Adding documents to vector store...")
            self.vector_store.add_documents(splits)
            
            # Record successful processing
            logger.info("Marking file as processed...")
            self.mark_file_processed(
                filepath=str(file_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                num_chunks=len(splits)
            )
            
            logger.info("File processing completed successfully")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing file {file_path}: {error_msg}")
            logger.exception("Full traceback:")
            
            # Record failed processing
            self.mark_file_processed(
                filepath=str(file_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                num_chunks=0,
                status='failed',
                error_message=error_msg
            )
            
            return False

    def get_all_documents(self) -> list:
        """
        Retrieve all documents from the vector store
        
        Returns:
            list: List of documents with their content and metadata
        """
        try:
            # Get all documents from the collection
            results = self.vector_store.get()
            
            # Format the results
            documents = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    doc = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def get_relevant_documents(self, query: str, k: int = 4) -> list:
        """
        Get relevant documents for a query without using the LLM
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            list: List of relevant documents with their content and similarity scores
        """
        try:
            # Get raw documents from the retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            )
            docs = retriever.get_relevant_documents(query)
            
            # Format the results
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                })
            return results
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {str(e)}")
            return []

    def get_direct_response(self, question: str) -> str:
        """
        Get response directly from LLM without using RAG context
        
        Args:
            question: Question to ask
            
        Returns:
            str: LLM's response
        """
        try:
            # Get response from selected model
            raw_output = self.get_llm_response(question)
            response_text = str(raw_output)
            
            # Log the direct query with all details
            self.log_query(
                query_type='direct',
                query_text=question,
                llm_input=question,  # For direct queries, input = question
                llm_output=response_text,
                final_response=response_text,
                retrieved_chunks=None
            )
            
            return response_text
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error getting direct response: {str(e)}")
            
            # Log error case
            self.log_query(
                query_type='direct',
                query_text=question,
                llm_input=question,
                llm_output=error_msg,
                final_response=error_msg,
                retrieved_chunks=None
            )
            return error_msg

    def log_query(self, query_type: str, query_text: str, llm_input: str, 
                  llm_output: str, final_response: str, retrieved_chunks: list = None):
        """Log query and response to database"""
        try:
            session = Session()
            log_entry = QueryLog(
                query_type=query_type,
                query_text=query_text,
                llm_input=llm_input,
                llm_output=llm_output,
                final_response=final_response,
                retrieved_chunks=json.dumps(retrieved_chunks) if retrieved_chunks else None
            )
            session.add(log_entry)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")

    def get_processing_history(self) -> list[dict]:
        """Get history of all processed files"""
        try:
            session = Session()
            processed_files = session.query(ProcessedFile).all()
            
            history = [{
                'filepath': file.filepath,
                'timestamp': file.timestamp,
                'chunk_size': file.chunk_size,
                'chunk_overlap': file.chunk_overlap,
                'num_chunks': file.num_chunks,
                'status': file.status,
                'error_message': file.error_message
            } for file in processed_files]
            
            session.close()
            return history
        except Exception as e:
            logger.error(f"Error getting processing history: {str(e)}")
            return []

    def index_all_files(self, documents_dir: str = "documents") -> bool:
        """Index all unprocessed files in the documents directory"""
        try:
            unprocessed = self.get_unprocessed_files(documents_dir)
            if not unprocessed:
                logger.info("No new files to process")
                return True
            
            for file_path in unprocessed:
                logger.info(f"Processing {file_path}")
                success = self.load_file(
                    file_path=file_path,
                    chunk_size=5000,
                    chunk_overlap=500,
                    metadata={"source": str(file_path)}
                )
                if not success:
                    logger.error(f"Failed to process {file_path}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error indexing files: {str(e)}")
            return False

    def clear_processed_files(self):
        """Clear all entries from the processed files table"""
        try:
            session = Session()
            session.query(ProcessedFile).delete()
            session.commit()
            session.close()
            # Also clear the vector store
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                persist_directory="data/chroma",
                embedding_function=self.embedding_model
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing processed files: {str(e)}")
            return False

    def remove_processed_file(self, filepath: str):
        """Remove a specific file from the processed files table"""
        try:
            session = Session()
            session.query(ProcessedFile).filter(ProcessedFile.filepath == filepath).delete()
            session.commit()
            session.close()
            # Note: We can't easily remove specific documents from Chroma,
            # so we'll just let the reindex overwrite the old embeddings
            return True
        except Exception as e:
            logger.error(f"Error removing processed file: {str(e)}")
            return False

    def get_document_chunks(self, filename: str) -> list[dict]:
        """Get all chunks for a specific document"""
        try:
            # Get all documents from the collection
            results = self.vector_store.get()
            
            # Filter and format chunks for the specific file
            chunks = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    metadata = results['metadatas'][i]
                    if metadata.get('source', '').endswith(filename):
                        chunks.append({
                            'id': results['ids'][i],
                            'content': results['documents'][i],
                            'metadata': metadata
                        })
            
            # Sort chunks by page number if available
            chunks.sort(key=lambda x: x['metadata'].get('page', 0))
            return chunks
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return []

    def export_document_chunks(self, filename: str, output_path: str = None) -> bool:
        """Export all chunks from a specific document to JSON"""
        try:
            chunks = self.get_document_chunks(filename)
            if not chunks:
                logger.error(f"No chunks found for {filename}")
                return False
            
            # Generate default output path if none provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"exports/chunks_{Path(filename).stem}_{timestamp}.json"
            
            # Ensure exports directory exists
            Path("exports").mkdir(exist_ok=True)
            
            # Create the export data structure
            export_data = {
                'filename': filename,
                'export_time': datetime.now().isoformat(),
                'chunk_count': len(chunks),
                'chunks': chunks
            }
            
            # Write to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully exported {len(chunks)} chunks to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting document chunks: {str(e)}")
            return False

