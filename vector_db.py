import faiss
import numpy as np
import sqlite3
import os

class PersistentVectorDatabase:
    def __init__(self, dimension, index_file='faiss_index.idx', db_file='documents.db'):
        self.dimension = dimension
        self.index_file = index_file
        self.db_file = db_file
        self.conn = None
        self.cursor = None

        # Initialize FAISS index
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        self.init_db()

    def init_db(self):
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS documents
                               (id INTEGER PRIMARY KEY, content TEXT, filename TEXT)''')
        self.conn.commit()

    def add_documents(self, embeddings, documents, filenames):
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        assert embeddings.shape[1] == self.dimension, f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}"
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Add to SQLite database
        for doc, filename in zip(documents, filenames):
            self.cursor.execute("INSERT INTO documents (content, filename) VALUES (?, ?)", (doc, filename))
        
        self.conn.commit()
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

    def search(self, query_embedding, k=5):
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        assert query_embedding.shape[1] == self.dimension, f"Query dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}"
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            self.cursor.execute("SELECT content FROM documents WHERE id=?", (int(idx)+1,))
            doc = self.cursor.fetchone()
            if doc:
                results.append((doc[0], distances[0][i]))
        
        return results

    def close_connection(self):
        if self.conn:
            self.conn.close()

    def __del__(self):
        self.close_connection()