from flask import Flask, render_template, request, jsonify
import json
import math
from typing import List, Dict, Any

app = Flask(__name__)

class LocalVectorDB:
    def __init__(self):
        """Initialize local vector database"""
        self.vectors = {}
        self.metadata = {}
        self.dimension = 128
        self.load_sample_data()
        
    def create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple word-based embedding locally"""
        words = text.lower().replace(",", " ").replace(".", " ").split()
        embedding = [0.0] * self.dimension
        
        keywords = {
            'engineer': [1.0, 0.8, 0.6, 0.4],
            'software': [0.9, 1.0, 0.7, 0.5],
            'programming': [0.8, 0.9, 1.0, 0.6],
            'python': [0.7, 0.8, 0.9, 1.0],
            'data': [0.6, 0.7, 0.8, 0.9],
            'scientist': [0.5, 0.6, 0.7, 0.8],
            'science': [0.4, 0.5, 0.6, 0.7],
            'web': [0.3, 0.4, 0.5, 0.6],
            'development': [0.2, 0.3, 0.4, 0.5],
            'design': [0.1, 0.2, 0.3, 0.4],
            'designer': [0.1, 0.2, 0.3, 0.4],
            'ux': [0.1, 0.2, 0.3, 0.4],
            'machine': [0.6, 0.7, 0.8, 0.9],
            'learning': [0.6, 0.7, 0.8, 0.9],
            'html': [0.3, 0.4, 0.5, 0.6],
            'css': [0.3, 0.4, 0.5, 0.6],
            'javascript': [0.3, 0.4, 0.5, 0.6],
            'manager': [0.2, 0.3, 0.4, 0.5],
            'product': [0.2, 0.3, 0.4, 0.5]
        }
        
        for i, word in enumerate(words[:20]):
            if word in keywords:
                features = keywords[word]
                start_idx = (i * 4) % (self.dimension - 4)
                for j, feature in enumerate(features):
                    if start_idx + j < self.dimension:
                        embedding[start_idx + j] += feature
        
        for i, word in enumerate(words[:10]):
            if word not in keywords and len(word) > 2:
                word_feature = (ord(word[0]) + ord(word[-1])) / 200.0
                idx = (hash(word) % (self.dimension - 20)) + 20
                if idx < self.dimension:
                    embedding[idx] = word_feature
        
        magnitude = sum(x*x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def load_sample_data(self):
        """Load sample data into the vector database"""
        # Sample structured data - REMOVED, we don't want person search
        # structured_data = [...]
        
        # Sample unstructured data
        unstructured_data = [
            {
                "id": "doc1",
                "title": "Python Programming",
                "content": "Python is a programming language used for software development web development data science machine learning"
            },
            {
                "id": "doc2", 
                "title": "Web Development",
                "content": "Web development involves HTML CSS JavaScript frameworks like React Vue Angular for building websites"
            },
            {
                "id": "doc3",
                "title": "Data Science",
                "content": "Data science uses statistics programming Python R SQL pandas numpy scikit-learn for analyzing data"
            },
            {
                "id": "doc4",
                "title": "User Experience Design",
                "content": "UX design focuses on user interface usability accessibility design thinking user research prototyping"
            },
            {
                "id": "doc5",
                "title": "Machine Learning",
                "content": "Machine learning algorithms neural networks deep learning artificial intelligence pattern recognition"
            }
        ]
        
        # Insert unstructured data only (no person search)
        for doc in unstructured_data:
            embedding = self.create_simple_embedding(doc['content'])
            
            vector_id = f"unstructured_{doc['id']}"
            self.vectors[vector_id] = embedding
            self.metadata[vector_id] = {
                "type": "unstructured",
                "doc_id": doc['id'],
                "title": doc['title'],
                "content_preview": doc['content'][:200],
                "content_length": len(doc['content'])
            }
    
    def search_similar(self, query: str, top_k: int = 5, filter_type: str = None):
        """Search for similar items in the local vector database"""
        query_embedding = self.create_simple_embedding(query)
        
        similarities = []
        for vector_id, vector in self.vectors.items():
            metadata = self.metadata[vector_id]
            
            if filter_type and metadata['type'] != filter_type:
                continue
            
            similarity = self.cosine_similarity(query_embedding, vector)
            similarities.append({
                'id': vector_id,
                'score': similarity,
                'metadata': metadata
            })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def get_all_data(self):
        """Get all data for display"""
        return {
            'vectors': len(self.vectors),
            'structured': len([v for v in self.metadata.values() if v['type'] == 'structured']),
            'unstructured': len([v for v in self.metadata.values() if v['type'] == 'unstructured']),
            'data': list(self.metadata.values())
        }

# Initialize the vector database
vector_db = LocalVectorDB()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint"""
    data = request.get_json()
    query = data.get('query', '')
    filter_type = data.get('filter', None)
    top_k = data.get('top_k', 5)
    
    if filter_type == 'all':
        filter_type = None
    
    results = vector_db.search_similar(query, top_k, filter_type)
    return jsonify(results)

@app.route('/data')
def get_data():
    """Get all data endpoint"""
    return jsonify(vector_db.get_all_data())

if __name__ == '__main__':
    import os
    print("🚀 Starting Vector Database Web Interface...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("💡 This is a completely local vector database running in your browser!")
    
    # Get port from environment variable (for deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
