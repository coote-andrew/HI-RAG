from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from langchain_rag import LangChainRAG
import logging
from logging.handlers import RotatingFileHandler
import os
import markdown2  # For converting markdown to HTML

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging - sets up both file and console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # File handler with rotation (1MB max size, keeping 10 backups)
        RotatingFileHandler(
            'logs/app.log',
            maxBytes=1024 * 1024,  # 1MB
            backupCount=10
        ),
        # Console handler for immediate feedback
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize Flask app with CORS (Cross-Origin Resource Sharing) enabled
app = Flask(__name__)
CORS(app)  # Allows the frontend to make requests to this API from a different domain/port

# Initialize RAG system with OpenAI as the LLM provider

rag = LangChainRAG(
    use_openai=True,
    openai_api_key=os.environ.get('openai_api_key')
)

@app.route('/api/chunks', methods=['GET'])
def get_chunks():
    try:
        # Retrieve all document chunks from the vector database
        documents = rag.get_all_documents()
        
        # Format the response with limited content for each chunk
        return jsonify({
            'status': 'success',
            'count': len(documents),
            'chunks': [{
                'id': i + 1,
                'content': doc['content'][:500],  # Limiting content length for response
                'metadata': doc['metadata']
            } for i, doc in enumerate(documents)]
        })
    except Exception as e:
        # Log any errors and return error response
        logger.error(f"Error getting chunks: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500  # HTTP 500 = Internal Server Error

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        # Extract data from the POST request
        question = request.json.get('question')
        conversation_id = request.json.get('conversation_id')
        previous_query = request.json.get('previous_query')
        previous_sql = request.json.get('previous_sql')
        previous_chat = request.json.get('previous_chat')
        
        logger.info(f"Received question: {question}")
        
        # Validate the question
        if not question:
            logger.warning("No question provided in request")
            return jsonify({
                'status': 'error',
                'message': 'No question provided'
            }), 400  # HTTP 400 = Bad Request
        
        # Process the question through the RAG system with conversation context
        try:
            result = rag.get_response(
                question, 
                conversation_id, 
                previous_query,
                previous_sql,
                previous_chat
            )
        except Exception as e:
            logger.error(f"Error in RAG response: {str(e)}", exc_info=True)
            raise
        
        # Convert markdown-formatted response to HTML for display
        html_response = markdown2.markdown(result['response'])
        
        logger.info("Successfully processed question")
        
        # Return a structured response with all the results
        return jsonify({
            'status': 'success',
            'response': html_response,             # The HTML-formatted response
            'query_text': result['query_text'],    # The refined health intelligence query
            'sql_query': result['sql_query'],      # Generated SQL query if any
            'conversation_id': result['conversation_id'],  # For maintaining conversation state
            'suggested_refinements': result.get('suggested_refinements', [])  # Optional refinement suggestions
        })
    except Exception as e:
        # Log and return any errors that occurred
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/direct', methods=['POST'])
def direct():
    try:
        # Get the question from the request
        question = request.json.get('question')
        
        # Validate the question
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'No question provided'
            }), 400
        
        # Get a direct response from the LLM (bypassing the retrieval)
        response = rag.get_direct_response(question)
        
        # Return the response
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        logger.error(f"Error processing direct question: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def history():
    try:
        # Get processing history from the RAG system
        history = rag.get_processing_history()
        
        # Return the processing history as JSON
        return jsonify({
            'status': 'success',
            'history': history
        })
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    try:
        # Get list of files that haven't been processed yet
        unprocessed = rag.get_unprocessed_pdfs()
        
        # Return status information
        return jsonify({
            'status': 'success',
            'unprocessed_files': unprocessed,
            'needs_indexing': len(unprocessed) > 0
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    # Render the main application HTML template
    return render_template('index.html')

@app.route('/pdf/<path:filename>')
def serve_pdf(filename):
    """Serve PDF files from the documents directory"""
    # Remove any 'documents/' prefix if it exists in the filename
    clean_filename = filename.replace('documents/', '')
    try:
        # Serve the requested PDF file
        return send_from_directory('documents', clean_filename)
    except Exception as e:
        logger.error(f"Error serving PDF {filename}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'PDF not found: {filename}'
        }), 404  # HTTP 404 = Not Found

if __name__ == '__main__':
    # Run the Flask application when executed directly
    app.run(debug=True, port=5000)