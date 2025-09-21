import os
import json
import math
import base64
from typing import List, Dict, Any
from PIL import Image
import numpy as np

class ImageVectorDB:
    def __init__(self):
        """Initialize local vector database with image support"""
        self.vectors = {}
        self.metadata = {}
        self.dimension = 128
        print("🖼️ Image Vector Database initialized!")
        
    def create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple word-based embedding for text"""
        words = text.lower().replace(",", " ").replace(".", " ").split()
        embedding = [0.0] * self.dimension
        
        keywords = {
            'engineer': [1.0, 0.8, 0.6, 0.4],
            'software': [0.9, 1.0, 0.7, 0.5],
            'programming': [0.8, 0.9, 1.0, 0.6],
            'python': [0.7, 0.8, 0.9, 1.0],
            'data': [0.6, 0.7, 0.8, 0.9],
            'cat': [0.9, 0.7, 0.5, 0.3],
            'dog': [0.8, 0.6, 0.4, 0.2],
            'animal': [0.7, 0.5, 0.3, 0.1],
            'nature': [0.6, 0.4, 0.8, 0.6],
            'landscape': [0.5, 0.3, 0.7, 0.5],
            'city': [0.4, 0.2, 0.6, 0.4],
            'building': [0.3, 0.1, 0.5, 0.3],
            'car': [0.2, 0.8, 0.4, 0.6],
            'vehicle': [0.1, 0.7, 0.3, 0.5],
            'food': [0.8, 0.3, 0.7, 0.1],
            'person': [0.7, 0.2, 0.6, 0.8],
            'face': [0.6, 0.1, 0.5, 0.7],
            'smile': [0.5, 0.9, 0.4, 0.6],
            'happy': [0.4, 0.8, 0.3, 0.5],
            'blue': [0.3, 0.7, 0.2, 0.4],
            'red': [0.2, 0.6, 0.1, 0.3],
            'green': [0.1, 0.5, 0.9, 0.2],
            'flower': [0.9, 0.4, 0.8, 0.7],
            'tree': [0.8, 0.3, 0.7, 0.6],
            'sky': [0.7, 0.2, 0.6, 0.5]
        }
        
        for i, word in enumerate(words[:20]):
            if word in keywords:
                features = keywords[word]
                start_idx = (i * 4) % (self.dimension - 4)
                for j, feature in enumerate(features):
                    if start_idx + j < self.dimension:
                        embedding[start_idx + j] += feature
        
        # Normalize the embedding
        magnitude = sum(x*x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def create_image_embedding(self, image_path: str, description: str = "") -> List[float]:
        """Create embedding for an image based on simple visual features and description"""
        try:
            # Load and process image
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Resize to standard size for processing
            image = image.resize((64, 64))
            img_array = np.array(image)
            
            # Extract simple visual features
            embedding = [0.0] * self.dimension
            
            # Color histogram features (RGB averages)
            r_avg = np.mean(img_array[:, :, 0]) / 255.0
            g_avg = np.mean(img_array[:, :, 1]) / 255.0
            b_avg = np.mean(img_array[:, :, 2]) / 255.0
            
            embedding[0] = r_avg
            embedding[1] = g_avg
            embedding[2] = b_avg
            
            # Brightness and contrast
            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0
            
            embedding[3] = brightness
            embedding[4] = contrast
            
            # Simple edge detection (gradient magnitude)
            gray = np.mean(img_array, axis=2)
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            edge_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2)) / 255.0
            
            embedding[5] = edge_magnitude
            
            # If description is provided, combine with text embedding
            if description:
                text_embedding = self.create_simple_embedding(description)
                # Combine image features with text features
                for i in range(6, min(len(text_embedding) + 6, self.dimension)):
                    if i - 6 < len(text_embedding):
                        embedding[i] = text_embedding[i - 6] * 0.7  # Weight text features
            
            # Fill remaining with image texture features
            for i in range(len(embedding)):
                if embedding[i] == 0 and i < 64:
                    # Simple texture feature based on pixel variance in small regions
                    row = (i // 8) * 8
                    col = (i % 8) * 8
                    if row + 8 <= 64 and col + 8 <= 64:
                        region = gray[row:row+8, col:col+8]
                        embedding[i] = np.var(region) / 10000.0  # Normalize variance
            
            # Normalize final embedding
            magnitude = sum(x*x for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
            
            return embedding
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Fallback to text-only embedding if image processing fails
            return self.create_simple_embedding(description)
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for web display"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except:
            return ""
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def insert_image_data(self, image_data: List[Dict[str, str]]):
        """Insert image data into the vector database"""
        for item in image_data:
            image_path = item['path']
            description = item.get('description', '')
            tags = item.get('tags', '')
            
            # Combine description and tags for better embedding
            full_text = f"{description} {tags}".strip()
            
            # Create embedding from image and text
            embedding = self.create_image_embedding(image_path, full_text)
            
            vector_id = f"image_{item['id']}"
            self.vectors[vector_id] = embedding
            
            # Convert image to base64 for storage
            image_base64 = self.image_to_base64(image_path)
            
            self.metadata[vector_id] = {
                "type": "image",
                "image_id": item['id'],
                "path": image_path,
                "description": description,
                "tags": tags,
                "image_data": image_base64,
                "filename": os.path.basename(image_path)
            }
        
        print(f"📸 Inserted {len(image_data)} images")
    
    def insert_text_data(self, text_data: List[Dict[str, str]]):
        """Insert text documents"""
        for doc in text_data:
            embedding = self.create_simple_embedding(doc['content'])
            
            vector_id = f"text_{doc['id']}"
            self.vectors[vector_id] = embedding
            self.metadata[vector_id] = {
                "type": "text",
                "doc_id": doc['id'],
                "title": doc['title'],
                "content": doc['content'][:200],
                "content_length": len(doc['content'])
            }
        
        print(f"📄 Inserted {len(text_data)} text documents")
    
    def search_similar(self, query: str, top_k: int = 5, filter_type: str = None):
        """Search for similar items"""
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
    
    def display_results(self, results: List[Dict], query: str):
        """Display search results with image support"""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 50)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            score = result['score']
            metadata = result['metadata']
            
            print(f"\n{i}. Similarity Score: {score:.3f}")
            print(f"   Type: {metadata['type']}")
            
            if metadata['type'] == 'image':
                print(f"   📸 Image: {metadata['filename']}")
                print(f"   Path: {metadata['path']}")
                print(f"   Description: {metadata['description']}")
                print(f"   Tags: {metadata['tags']}")
            else:
                print(f"   📄 Document: {metadata['title']}")
                print(f"   Content: {metadata['content']}...")
            
            print("-" * 30)

def create_sample_data():
    """Create sample data with images and text"""
    
    # Sample image data (you would replace these paths with actual image files)
    image_data = [
        {
            "id": "img1",
            "path": "sample_images/cat.jpg",  # You need to add actual images
            "description": "A cute orange cat sitting on a windowsill",
            "tags": "cat animal pet orange cute"
        },
        {
            "id": "img2", 
            "path": "sample_images/landscape.jpg",
            "description": "Beautiful mountain landscape with blue sky",
            "tags": "nature landscape mountain sky blue scenery"
        },
        {
            "id": "img3",
            "path": "sample_images/city.jpg", 
            "description": "Modern city skyline at night with lights",
            "tags": "city urban skyline night lights building"
        },
        {
            "id": "img4",
            "path": "sample_images/flower.jpg",
            "description": "Red roses in a garden",
            "tags": "flower rose red garden nature bloom"
        }
    ]
    
    # Sample text data
    text_data = [
        {
            "id": "doc1",
            "title": "Pet Care Guide", 
            "content": "Taking care of cats and dogs requires regular feeding, grooming, and veterinary checkups. Cats are independent animals that need clean litter boxes."
        },
        {
            "id": "doc2",
            "title": "Photography Tips",
            "content": "Landscape photography works best during golden hour. Capture the natural beauty of mountains, forests, and skies with proper lighting."
        },
        {
            "id": "doc3",
            "title": "Urban Planning",
            "content": "Modern cities need efficient transportation, green spaces, and sustainable building designs to accommodate growing populations."
        }
    ]
    
    return image_data, text_data

def main():
    """Demo with image and text search"""
    print("🚀 Starting Image + Text Vector Database Demo")
    print("🖼️ Search for images and text using keywords!")
    print("=" * 60)
    
    # Initialize database
    db = ImageVectorDB()
    
    # Load sample data
    image_data, text_data = create_sample_data()
    
    print("\n📸 Note: To use images, place sample images in 'sample_images/' folder")
    print("   For demo purposes, we'll use text descriptions only\n")
    
    # Insert text data (images would need actual files)
    db.insert_text_data(text_data)
    
    # Demo searches
    queries = [
        "cat animal",
        "mountain landscape", 
        "city lights",
        "flower garden",
        "pet care"
    ]
    
    for query in queries:
        results = db.search_similar(query, top_k=3)
        db.display_results(results, query)
    
    print("\n💡 To add real images:")
    print("1. Create 'sample_images/' folder")
    print("2. Add your image files (jpg, png)")
    print("3. Update image_data paths")
    print("4. Install Pillow: pip install Pillow numpy")
    print("5. Run again to search images by description!")

if __name__ == "__main__":
    main()
