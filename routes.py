from flask import Blueprint, request, jsonify
from .models.langchain_rag import LangChainRAG

main = Blueprint('main', __name__)
rag = LangChainRAG()  # Initialize once at startup

@main.route('/api/query', methods=['POST'])
def query():
    try:
        question = request.json.get('question')
        result = rag.get_response(question)
        return jsonify({
            'status': 'success',
            'response': result['response'],
            'sources': result['sources']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500