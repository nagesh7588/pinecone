import os
import json
import random
import hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

class PineconeVectorDB:
    def __init__(self):
        """Initialize Pinecone client (no OpenAI needed!)"""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Index configuration - using smaller dimension for simplicity
        self.index_name = "simple-demo-v2"  # New index name
        self.dimension = 128  # Much smaller dimension for simple demo
        
        # Create or connect to index
        self.setup_index()
    
    def setup_index(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                print(f"Creating index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print("Index created successfully!")
            else:
                print(f"Index '{self.index_name}' already exists.")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to index '{self.index_name}'")
            
        except Exception as e:
            print(f"Error setting up index: {e}")
            raise
    
    def create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple word-based embedding (no OpenAI needed!)"""
        # Convert text to lowercase and split into words
        words = text.lower().replace(",", " ").replace(".", " ").split()
        
        # Initialize embedding vector
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
        for i, word in enumerate(words[:20]):  # Only consider first 20 words
            if word in keywords:
                features = keywords[word]
                start_idx = (i * 4) % (self.dimension - 4)
                for j, feature in enumerate(features):
                    if start_idx + j < self.dimension:
                        embedding[start_idx + j] += feature
        
        # Add character-based features for words not in keyword list
        for i, word in enumerate(words[:10]):
            if word not in keywords and len(word) > 2:
                # Use word length and first/last characters
                word_feature = (ord(word[0]) + ord(word[-1])) / 200.0
                idx = (hash(word) % (self.dimension - 20)) + 20
                if idx < self.dimension:
                    embedding[idx] = word_feature
        
        # Normalize the embedding
        magnitude = sum(x*x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def create_feature_embedding(self, features: Dict[str, Any]) -> List[float]:
        """Create embedding from structured features"""
        # Convert structured data to text and use word-based embedding
        text = f"{features['name']} {features['occupation']} age {features['age']}"
        return self.create_simple_embedding(text)
    
    def insert_structured_data(self, data: List[Dict[str, Any]]):
        """Insert structured data (like table rows) into Pinecone"""
        vectors = []
        
        for item in data:
            # Create embedding from structured features
            embedding = self.create_feature_embedding(item)
            
            # Prepare vector for upsert
            vector = {
                "id": f"structured_{item['id']}",
                "values": embedding,
                "metadata": {
                    "type": "structured",
                    "user_id": item['id'],
                    "name": item['name'],
                    "age": item['age'],
                    "occupation": item['occupation'],
                    "text_representation": f"ID: {item['id']}, Name: {item['name']}, Age: {item['age']}, Occupation: {item['occupation']}"
                }
            }
            vectors.append(vector)
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        print(f"Inserted {len(vectors)} structured records")
    
    def insert_unstructured_data(self, documents: List[Dict[str, str]]):
        """Insert unstructured data (text documents) into Pinecone"""
        vectors = []
        
        for doc in documents:
            # Create simple embedding for the document text
            embedding = self.create_simple_embedding(doc['content'])
            
            # Prepare vector for upsert
            vector = {
                "id": f"unstructured_{doc['id']}",
                "values": embedding,
                "metadata": {
                    "type": "unstructured",
                    "doc_id": doc['id'],
                    "title": doc['title'],
                    "content_preview": doc['content'][:200],  # Store first 200 chars in metadata
                    "content_length": len(doc['content'])
                }
            }
            vectors.append(vector)
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        print(f"Inserted {len(vectors)} unstructured documents")
    
    def search_similar(self, query: str, top_k: int = 5, filter_type: str = None, is_structured: bool = False):
        """Search for similar items in the vector database"""
        # Create embedding for the query
        if is_structured:
            # For structured queries, try to parse as features
            query_embedding = self.create_simple_embedding(query)
        else:
            query_embedding = self.create_simple_embedding(query)
        
        # Prepare filter if specified
        filter_dict = {"type": filter_type} if filter_type else None
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return results
    
    def print_search_results(self, results, query: str):
        """Pretty print search results"""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 50)
        
        if not results.matches:
            print("No results found.")
            return
        
        for i, match in enumerate(results.matches, 1):
            score = match.score
            metadata = match.metadata
            
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

def main():
    """Main function to demonstrate the vector database"""
    print("🚀 Starting Simple Pinecone Vector Database Demo")
    print("📝 No OpenAI needed - using simple hash-based embeddings!")
    print("=" * 60)
    
    try:
        # Initialize the vector database
        vector_db = PineconeVectorDB()
        
        # Sample structured data (like database rows)
        structured_data = [
            {"id": 1, "name": "Alice Johnson", "age": 28, "occupation": "Software Engineer"},
            {"id": 2, "name": "Bob Smith", "age": 35, "occupation": "Data Scientist"},
            {"id": 3, "name": "Carol Davis", "age": 42, "occupation": "Product Manager"},
            {"id": 4, "name": "David Wilson", "age": 31, "occupation": "UX Designer"},
            {"id": 5, "name": "Eva Brown", "age": 29, "occupation": "Machine Learning Engineer"},
            {"id": 6, "name": "Frank Miller", "age": 45, "occupation": "Software Engineer"},
            {"id": 7, "name": "Grace Lee", "age": 33, "occupation": "Data Scientist"}
        ]
        
        # Sample unstructured data (text documents)
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
        
        # Insert data into Pinecone
        print("\n📥 Inserting structured data...")
        vector_db.insert_structured_data(structured_data)
        
        print("\n📥 Inserting unstructured data...")
        vector_db.insert_unstructured_data(unstructured_data)
        
        print("\n✅ Data insertion completed!")
        
        # Wait a moment for indexing
        import time
        print("\n⏳ Waiting for indexing to complete...")
        time.sleep(3)
        
        # Example searches with simple queries
        search_queries = [
            ("engineer", "Looking for engineers"),
            ("programming", "Looking for programming content"),
            ("data", "Looking for data-related content"),
            ("design", "Looking for design content"),
            ("python", "Looking for Python content")
        ]
        
        print("\n🔎 Performing similarity searches...")
        
        for query, description in search_queries:
            print(f"\n🔍 {description}: '{query}'")
            results = vector_db.search_similar(query, top_k=3)
            vector_db.print_search_results(results, query)
        
        # Search only structured data
        print("\n🔍 Searching only structured data for 'engineer':")
        results = vector_db.search_similar("engineer", top_k=3, filter_type="structured")
        vector_db.print_search_results(results, "engineer (structured only)")
        
        # Search only unstructured data
        print("\n🔍 Searching only unstructured data for 'programming':")
        results = vector_db.search_similar("programming", top_k=3, filter_type="unstructured")
        vector_db.print_search_results(results, "programming (unstructured only)")
        
        print("\n🎉 Demo completed successfully!")
        print("💡 This demo uses simple hash-based embeddings - no external APIs needed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your .env file contains a valid Pinecone API key:")
        print("- PINECONE_API_KEY")
        print("- PINECONE_ENVIRONMENT")

if __name__ == "__main__":
    main()
