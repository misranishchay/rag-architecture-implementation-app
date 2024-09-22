import PyPDF2
from docx import Document
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

def read_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    elif file_extension.lower() == '.docx':
        doc = Document(file_path)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def split_into_chunks(text, chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for the space
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    
    return np.array(embeddings)



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_answer(query, context, api_key):
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Context: {context}
    
    Human: {query}
    
    Assistant: Based on the context provided, I'll do my best to answer the question. If the answer isn't in the context, I'll say so.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        raise

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import os
import logging
from utils import read_document, split_into_chunks, get_embeddings

def process_documents(document_dir, db):
    for filename in os.listdir(document_dir):
        file_path = os.path.join(document_dir, filename)
        if os.path.isfile(file_path):
            try:
                logging.info(f"Processing file: {filename}")
                
                # Check if file has already been processed
                db.cursor.execute("SELECT id FROM documents WHERE filename=?", (filename,))
                if db.cursor.fetchone():
                    logging.info(f"File {filename} already processed. Skipping.")
                    continue
                
                # Read the document
                text = read_document(file_path)
                if not text.strip():
                    logging.warning(f"Empty document {filename}. Skipping.")
                    continue
                
                # Split into chunks
                chunks = split_into_chunks(text)
                if not chunks:
                    logging.warning(f"No chunks generated for {filename}. Skipping.")
                    continue
                
                # Generate embeddings
                embeddings = get_embeddings(chunks)
                if len(embeddings) == 0:
                    logging.warning(f"No valid embeddings generated for {filename}. Skipping.")
                    continue
                
                # Add to database
                db.add_documents(embeddings, chunks, [filename] * len(chunks))
                logging.info(f"Successfully processed {filename}")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

    logging.info("Document processing complete.")