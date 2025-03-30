from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class QueryLog(Base):
    __tablename__ = 'query_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query_type = Column(String(10))  # 'rag' or 'direct'
    query_text = Column(Text)        # Original user query
    llm_input = Column(Text)         # What was actually sent to LLM (including context for RAG)
    llm_output = Column(Text)        # Raw LLM output
    final_response = Column(Text)     # Processed/formatted response sent to user
    retrieved_chunks = Column(Text, nullable=True)  # Retrieved documents for RAG

class ProcessedFile(Base):
    __tablename__ = 'processed_files'
    
    id = Column(Integer, primary_key=True)
    filepath = Column(String, unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)
    num_chunks = Column(Integer)
    status = Column(String)  # 'success' or 'failed'
    error_message = Column(Text, nullable=True)

class ConversationHistory(Base):
    __tablename__ = 'conversation_history'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(36))  # UUID
    timestamp = Column(DateTime, default=datetime.utcnow)
    question = Column(Text)
    response = Column(Text)
    query_text = Column(Text)
    sources = Column(Text)  # JSON string of sources
    refinements = Column(Text)  # JSON string of suggested refinements


# Create engine and tables
engine = create_engine('sqlite:///data/query_logs.db')
Base.metadata.create_all(engine)

# Create session factory
Session = sessionmaker(bind=engine) 