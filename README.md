## Project Documentation


# Document Q&A System 

## Table of Contents
1. Project Overview
2. System Architecture
3. Component Breakdown
   3.1 Flask Application (app.py)
   3.2 Vector Database (vector_db.py)
   3.3 Utility Functions (utils.py)
4. API Endpoints
5. Data Flow
6. Setup and Installation
7. Usage Guide
8. Extending the System
9. Security Considerations
10. Troubleshooting

## 1. Project Overview

The Document Q&A System is a web-based application that allows users to upload documents, process them, and ask questions about their content. 
The system uses natural language processing and machine learning techniques to understand the documents and generate relevant answers to user queries.

Key Features:
- Document upload and processing
- Question answering based on document content
- Vector-based document storage for efficient retrieval
- Integration with OpenAI's language models for answer generation

## 2. System Architecture

The system follows a simple client-server architecture:

- Frontend: HTML-based web interface
- Backend: Flask web server
- Storage: SQLite database for document metadata, FAISS for vector storage
- External Services: OpenAI API for natural language processing

## 3. Component Breakdown

### 3.1 Flask Application (app.py)

The main application file that sets up the web server and defines the API endpoints.

Key Functions:
- `create_app()`: Initializes the Flask application and sets up routes
- `upload_file()`: Handles document uploads
- `ask_question()`: Processes user questions and returns answers

### 3.2 Vector Database (vector_db.py)

Manages the storage and retrieval of document embeddings.

Key Class:
- `PersistentVectorDatabase`: Handles FAISS index and SQLite database operations

Key Methods:
- `init_db()`: Initializes the SQLite database
- `add_documents()`: Adds new documents to the FAISS index and SQLite database
- `search()`: Searches for similar documents given a query embedding

### 3.3 Utility Functions (utils.py)

Contains helper functions for document processing and embedding generation.

Key Functions:
- `read_document()`: Extracts text from PDF and DOCX files
- `split_into_chunks()`: Divides document text into manageable chunks
- `get_embeddings()`: Generates embeddings for text chunks
- `generate_answer()`: Uses OpenAI API to generate answers based on context

## 4. API Endpoints

1. `/upload` (POST)
   - Purpose: Upload and process new documents
   - Parameters: File in the request body
   - Response: Success message or error

2. `/ask` (POST)
   - Purpose: Answer questions based on uploaded documents
   - Parameters: 'question' in the request body
   - Response: Generated answer or error message

## 5. Data Flow

1. Document Upload:
   User -> Frontend -> `/upload` endpoint -> `read_document()` -> `split_into_chunks()` -> `get_embeddings()` -> `PersistentVectorDatabase.add_documents()`

2. Question Answering:
   User -> Frontend -> `/ask` endpoint -> `get_embeddings()` -> `PersistentVectorDatabase.search()` -> `generate_answer()` -> Frontend -> User

## 6. Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up OpenAI API key as an environment variable
4. Run the application: `python app.py`

## 7. Usage Guide

1. Access the web interface at `http://localhost:5000`
2. Use the "Upload Document" form to add new documents
3. Use the "Ask a Question" form to query the system
4. View the generated answers on the same page

## 8. Extending the System

- Add support for more document types in `read_document()`
- Implement more sophisticated chunking strategies in `split_into_chunks()`
- Explore different embedding models in `get_embeddings()`
- Enhance the frontend with JavaScript for a more dynamic user experience

## 9. Security Considerations

- The OpenAI API key is currently set in the code. In a production environment, use environment variables or a secure secret management system.
- Implement user authentication and authorization for multi-user scenarios
- Validate and sanitize all user inputs to prevent injection attacks
- Use HTTPS in production to encrypt data in transit

## 10. Troubleshooting

- Check logs for detailed error messages
- Ensure all dependencies are correctly installed
- Verify that the OpenAI API key is correctly set and valid
- Check file permissions for the SQLite database and FAISS index files

For any questions , please reachout to - misranishchay@icloud.com
