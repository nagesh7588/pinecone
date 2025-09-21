import os
import json
import math
from typing import List, Dict, Any

class LocalVectorDB:
    def __init__(self):
        """Initialize local vector database (no external APIs!)"""
        self.vectors = {}  # Store vectors in memory
        self.metadata = {}  # Store metadata
        self.dimension = 128
        print("🏠 Local Vector Database initialized!")
        
    def create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple word-based embedding locally"""
        words = text.lower().replace(",", " ").replace(".", " ").split()
        embedding = [0.0] * self.dimension
        
        # Simple keyword-based features
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
        
        # Assign features based on keywords found
        for i, word in enumerate(words[:20]):
            if word in keywords:
                features = keywords[word]
                start_idx = (i * 4) % (self.dimension - 4)
                for j, feature in enumerate(features):
                    if start_idx + j < self.dimension:
                        embedding[start_idx + j] += feature
        
        # Add character-based features for words not in keyword list
        for i, word in enumerate(words[:10]):
            if word not in keywords and len(word) > 2:
                word_feature = (ord(word[0]) + ord(word[-1])) / 200.0
                idx = (hash(word) % (self.dimension - 20)) + 20
                if idx < self.dimension:
                    embedding[idx] = word_feature
        
        # Normalize the embedding
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
    
    def insert_structured_data(self, data: List[Dict[str, Any]]):
        """Insert structured data into local vector database"""
        for item in data:
            text = f"{item['name']} {item['occupation']} age {item['age']}"
            embedding = self.create_simple_embedding(text)
            
            vector_id = f"structured_{item['id']}"
            self.vectors[vector_id] = embedding
            self.metadata[vector_id] = {
                "type": "structured",
                "user_id": item['id'],
                "name": item['name'],
                "age": item['age'],
                "occupation": item['occupation']
            }
        
        print(f"📥 Inserted {len(data)} structured records locally")
    
    def insert_unstructured_data(self, documents: List[Dict[str, str]]):
        """Insert unstructured data into local vector database"""
        for doc in documents:
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
        
        print(f"📥 Inserted {len(documents)} unstructured documents locally")
    
    def search_similar(self, query: str, top_k: int = 5, filter_type: str = None):
        """Search for similar items in the local vector database"""
        query_embedding = self.create_simple_embedding(query)
        
        # Calculate similarities
        similarities = []
        for vector_id, vector in self.vectors.items():
            metadata = self.metadata[vector_id]
            
            # Apply filter if specified
            if filter_type and metadata['type'] != filter_type:
                continue
            
            similarity = self.cosine_similarity(query_embedding, vector)
            similarities.append({
                'id': vector_id,
                'score': similarity,
                'metadata': metadata
            })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def print_search_results(self, results: List[Dict], query: str):
        """Pretty print search results"""
        print(f"\n🔍 Local Search Results for: '{query}'")
        print("=" * 50)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            score = result['score']
            metadata = result['metadata']
            
            print(f"\n{i}. Similarity Score: {score:.3f}")
            print(f"   Type: {metadata['type']}")
            
            if metadata['type'] == 'structured':
                print(f"   ID: {metadata['user_id']}")
                print(f"   Name: {metadata['name']}")
                print(f"   Age: {metadata['age']}")
                print(f"   Occupation: {metadata['occupation']}")
            else:
                print(f"   Document ID: {metadata['doc_id']}")
                print(f"   Title: {metadata['title']}")
                print(f"   Content Preview: {metadata['content_preview']}...")
                print(f"   Content Length: {metadata['content_length']} chars")
            
            print("-" * 30)
    
    def save_to_file(self, filename: str = "local_vectors.json"):
        """Save the vector database to a local file"""
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Database saved to {filename}")
    
    def load_from_file(self, filename: str = "local_vectors.json"):
        """Load the vector database from a local file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.vectors = data["vectors"]
            self.metadata = data["metadata"]
            print(f"📂 Database loaded from {filename}")
        except FileNotFoundError:
            print(f"❌ File {filename} not found")

def main():
    """Main function to demonstrate the local vector database"""
    print("🚀 Starting LOCAL Vector Database Demo")
    print("🏠 Everything runs on your computer - no cloud needed!")
    print("=" * 60)
    
    # Initialize the local vector database
    vector_db = LocalVectorDB()
    
    # Sample structured data
    structured_data = [
        {"id": 1, "name": "Alice Johnson", "age": 28, "occupation": "Software Engineer"},
        {"id": 2, "name": "Bob Smith", "age": 35, "occupation": "Data Scientist"},
        {"id": 3, "name": "Carol Davis", "age": 42, "occupation": "Product Manager"},
        {"id": 4, "name": "David Wilson", "age": 31, "occupation": "UX Designer"},
        {"id": 5, "name": "Eva Brown", "age": 29, "occupation": "Machine Learning Engineer"},
        {"id": 6, "name": "Frank Miller", "age": 45, "occupation": "Software Engineer"},
        {"id": 7, "name": "Grace Lee", "age": 33, "occupation": "Data Scientist"}
    ]
    
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
    
    # Insert data
    vector_db.insert_structured_data(structured_data)
    vector_db.insert_unstructured_data(unstructured_data)
    
    print("\n✅ Data insertion completed!")
    
    # Save to local file
    vector_db.save_to_file()
    
    # Example searches
    search_queries = [
        ("engineer", "Looking for engineers"),
        ("programming", "Looking for programming content"),
        ("data", "Looking for data-related content"),
        ("design", "Looking for design content"),
        ("python", "Looking for Python content")
    ]
    
    print("\n🔎 Performing local similarity searches...")
    
    for query, description in search_queries:
        print(f"\n🔍 {description}: '{query}'")
        results = vector_db.search_similar(query, top_k=3)
        vector_db.print_search_results(results, query)
    
    # Filtered searches
    print("\n🔍 Searching only structured data for 'engineer':")
    results = vector_db.search_similar("engineer", top_k=3, filter_type="structured")
    vector_db.print_search_results(results, "engineer (structured only)")
    
    print("\n🔍 Searching only unstructured data for 'programming':")
    results = vector_db.search_similar("programming", top_k=3, filter_type="unstructured")
    vector_db.print_search_results(results, "programming (unstructured only)")
    
    print("\n🎉 Local demo completed successfully!")
    print("💡 All data stored locally in memory and saved to 'local_vectors.json'")
    print("🏠 No external APIs or cloud services needed!")

if __name__ == "__main__":
    main()
