import os
import sys
import logging
from flask import Flask, request, render_template_string, g
from werkzeug.utils import secure_filename
from vector_db import PersistentVectorDatabase
from utils import process_documents, get_embeddings, generate_answer
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def create_app():
    # Initialize Flask app
    app = Flask(__name__)

    # Configure upload folder
    UPLOAD_FOLDER = 'uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Initialize database
    try:
        db = PersistentVectorDatabase(384)  # 384 is the dimension for the all-MiniLM-L6-v2 model
    except Exception as e:
        logger.error(f"Failed to initialize PersistentVectorDatabase: {e}")
        sys.exit(1)

    # Set OpenAI API key (NOT RECOMMENDED for production use)
    OPENAI_API_KEY = 'XXXXXX-XXXXX-XXXX'  # Replace with your actual API key
    logger.warning("API key is set directly in the code. This is not recommended for production use.")

    if not OPENAI_API_KEY:
        logger.error("No OpenAI API key found. Please set the OPENAI_API_KEY.")
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # HTML template
    HTML_TEMPLATE = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Q&A System</title>
    </head>
    <body>
        <h1>Document Q&A System</h1>
        <h2>Upload Document</h2>
        <form method="post" enctype="multipart/form-data" action="/upload">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
        <h2>Ask a Question</h2>
        <form method="post" action="/ask">
            <input type="text" name="question" size="50">
            <input type="submit" value="Ask">
        </form>
        {% if answer %}
        <h2>Answer</h2>
        <p>{{ answer }}</p>
        {% endif %}
    </body>
    </html>
    '''

    @app.before_request
    def before_request():
        db.init_db()

    @app.teardown_appcontext
    def close_connection(exception):
        db.close_connection()

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                process_documents(app.config['UPLOAD_FOLDER'], db)
                return 'File uploaded and processed successfully'
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                return 'Error processing file', 500
            

    @app.route('/ask', methods=['POST'])
    def ask_question():
        question = request.form['question']
        logger.info(f"Received question: {question}")
        
        try:
            query_embedding = get_embeddings([question])[0]
            logger.debug(f"Generated query embedding of shape: {query_embedding.shape}")
            
            results = db.search(query_embedding, k=3)
            logger.debug(f"Search returned {len(results)} results")
            
            context = "\n".join([doc for doc, _ in results])
            logger.debug(f"Context (first 100 chars): {context[:100]}...")
            
            answer = generate_answer(question, context, OPENAI_API_KEY)
            logger.info(f"Generated answer: {answer}")
            
            return render_template_string(HTML_TEMPLATE, answer=answer)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return 'Error processing question', 500

    return app

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app = create_app()
    app.run(debug=True, threaded=True)