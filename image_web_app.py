from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import math
import base64
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ImageVectorDB:
    def __init__(self):
        self.vectors = {}
        self.metadata = {}
        self.dimension = 128
        self.load_sample_data()
        self.load_existing_images()  # Auto-load any existing images
        
    def create_simple_embedding(self, text: str) -> List[float]:
        """Create embedding from text"""
        words = text.lower().replace(",", " ").replace(".", " ").split()
        embedding = [0.0] * self.dimension
        
        keywords = {
            'cat': [1.0, 0.8, 0.6, 0.4], 'dog': [0.9, 0.7, 0.5, 0.3],
            'animal': [0.8, 0.6, 0.4, 0.2], 'pet': [0.7, 0.5, 0.3, 0.1],
            'nature': [0.6, 0.4, 0.8, 0.6], 'landscape': [0.5, 0.3, 0.7, 0.5],
            'mountain': [0.4, 0.2, 0.6, 0.4], 'sky': [0.3, 0.1, 0.5, 0.3],
            'city': [0.2, 0.8, 0.4, 0.6], 'building': [0.1, 0.7, 0.3, 0.5],
            'car': [0.8, 0.3, 0.7, 0.1], 'vehicle': [0.7, 0.2, 0.6, 0.8],
            'food': [0.6, 0.1, 0.5, 0.7], 'person': [0.5, 0.9, 0.4, 0.6],
            'face': [0.4, 0.8, 0.3, 0.5], 'smile': [0.3, 0.7, 0.2, 0.4],
            'flower': [0.2, 0.6, 0.1, 0.3], 'tree': [0.1, 0.5, 0.9, 0.2],
            'water': [0.9, 0.4, 0.8, 0.7], 'ocean': [0.8, 0.3, 0.7, 0.6],
            'beach': [0.7, 0.2, 0.6, 0.5], 'sunset': [0.6, 0.1, 0.5, 0.4],
            'blue': [0.5, 0.9, 0.4, 0.3], 'red': [0.4, 0.8, 0.3, 0.2],
            'green': [0.3, 0.7, 0.2, 0.1], 'yellow': [0.2, 0.6, 0.1, 0.9],
            'house': [0.1, 0.5, 0.9, 0.8], 'home': [0.9, 0.4, 0.8, 0.7],
            'garden': [0.8, 0.3, 0.7, 0.6], 'park': [0.7, 0.2, 0.6, 0.5]
        }
        
        for i, word in enumerate(words[:20]):
            if word in keywords:
                features = keywords[word]
                start_idx = (i * 4) % (self.dimension - 4)
                for j, feature in enumerate(features):
                    if start_idx + j < self.dimension:
                        embedding[start_idx + j] += feature
        
        # Add character-based features for unknown words
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
    
    def _auto_analyze_image(self, img_array: np.ndarray) -> str:
        """Automatically analyze image content to generate description"""
        try:
            # Color analysis
            r_avg = np.mean(img_array[:, :, 0])
            g_avg = np.mean(img_array[:, :, 1])
            b_avg = np.mean(img_array[:, :, 2])
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Determine dominant colors
            colors = []
            if r_avg > g_avg and r_avg > b_avg:
                if r_avg > 150:
                    colors.append("red")
            if g_avg > r_avg and g_avg > b_avg:
                if g_avg > 150:
                    colors.append("green")
            if b_avg > r_avg and b_avg > g_avg:
                if b_avg > 150:
                    colors.append("blue")
            
            # Brightness analysis
            if brightness > 200:
                colors.append("bright")
            elif brightness < 100:
                colors.append("dark")
            
            # Color combinations
            if r_avg > 150 and g_avg > 150 and b_avg < 100:
                colors.append("yellow")
            elif r_avg > 150 and g_avg < 100 and b_avg > 150:
                colors.append("purple")
            elif r_avg < 100 and g_avg > 150 and b_avg > 150:
                colors.append("cyan")
            elif r_avg > 150 and g_avg > 100 and b_avg < 100:
                colors.append("orange")
            
            # Texture analysis
            gray = np.mean(img_array, axis=2)
            edge_intensity = np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1)))
            
            textures = []
            if edge_intensity > 20:
                textures.append("detailed")
            elif edge_intensity < 5:
                textures.append("smooth")
            
            if contrast > 50:
                textures.append("high contrast")
            elif contrast < 20:
                textures.append("soft")
            
            # Combine analysis into description
            description_parts = []
            
            if colors:
                description_parts.append(" ".join(colors[:2]))  # Top 2 colors
            
            if textures:
                description_parts.append(" ".join(textures[:2]))  # Top 2 textures
            
            # Add common image types based on analysis
            if "green" in colors and brightness > 120:
                description_parts.append("nature outdoor plant")
            elif "blue" in colors and brightness > 150:
                description_parts.append("sky water outdoor")
            elif "dark" in colors and contrast > 40:
                description_parts.append("indoor shadow")
            elif brightness > 180 and contrast < 30:
                description_parts.append("light background")
            
            # Generate final description
            if description_parts:
                description = " ".join(description_parts)
            else:
                # Fallback based on overall characteristics
                if brightness > 150:
                    description = "bright colorful image"
                elif brightness < 100:
                    description = "dark image"
                else:
                    description = "image photo"
            
            return description.strip()
            
        except Exception as e:
            print(f"Error in auto-analysis: {e}")
            return "image photo"

    def create_image_embedding(self, image_path: str, description: str = "") -> List[float]:
        """Create more accurate embedding for image + description with auto-analysis"""
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize((64, 64))
            img_array = np.array(image)
            
            embedding = [0.0] * self.dimension
            
            # Auto-generate description if none provided
            if not description.strip():
                description = self._auto_analyze_image(img_array)
                print(f"🤖 Auto-analyzed: {description}")
            
            # If description is available (manual or auto), weight it heavily (80% of embedding)
            if description.strip():
                text_embedding = self.create_simple_embedding(description)
                # Use most of the embedding space for description
                for i in range(min(len(text_embedding), int(self.dimension * 0.8))):
                    embedding[i] = text_embedding[i]
                
                # Add some basic visual features (20% of embedding)
                visual_start = int(self.dimension * 0.8)
                r_avg = np.mean(img_array[:, :, 0]) / 255.0
                g_avg = np.mean(img_array[:, :, 1]) / 255.0
                b_avg = np.mean(img_array[:, :, 2]) / 255.0
                brightness = np.mean(img_array) / 255.0
                
                if visual_start < self.dimension:
                    embedding[visual_start] = r_avg
                if visual_start + 1 < self.dimension:
                    embedding[visual_start + 1] = g_avg
                if visual_start + 2 < self.dimension:
                    embedding[visual_start + 2] = b_avg
                if visual_start + 3 < self.dimension:
                    embedding[visual_start + 3] = brightness
            else:
                # Fallback: rely more on visual features (but less accurate)
                # Color histogram
                r_avg = np.mean(img_array[:, :, 0]) / 255.0
                g_avg = np.mean(img_array[:, :, 1]) / 255.0
                b_avg = np.mean(img_array[:, :, 2]) / 255.0
                
                embedding[0] = r_avg
                embedding[1] = g_avg  
                embedding[2] = b_avg
                
                # Brightness and texture
                brightness = np.mean(img_array) / 255.0
                contrast = np.std(img_array) / 255.0
                
                embedding[3] = brightness
                embedding[4] = contrast
                
                # Simple texture analysis
                gray = np.mean(img_array, axis=2)
                for i in range(5, min(25, self.dimension)):
                    row = ((i-5) // 4) * 16
                    col = ((i-5) % 4) * 16
                    if row + 16 <= 64 and col + 16 <= 64:
                        region = gray[row:row+16, col:col+16]
                        embedding[i] = np.var(region) / 10000.0
            
            # Normalize
            magnitude = sum(x*x for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
            
            return embedding
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # Return a basic embedding
            return [0.01] * self.dimension
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except:
            return ""
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def load_sample_data(self):
        """Load sample data"""
        # Sample text data that describes images
        sample_data = [
            {
                "id": "sample1",
                "type": "text", 
                "title": "Mountain Landscape",
                "content": "Beautiful mountain landscape with snow-capped peaks and blue sky nature scenery"
            },
            {
                "id": "sample2",
                "type": "text",
                "title": "City at Night", 
                "content": "Urban city skyline at night with bright lights and tall buildings"
            },
            {
                "id": "sample3",
                "type": "text",
                "title": "Cat Portrait",
                "content": "Cute orange cat sitting by the window pet animal feline"
            },
            {
                "id": "sample4",
                "type": "text",
                "title": "Flower Garden",
                "content": "Colorful flowers in a garden with roses and tulips nature bloom"
            },
            {
                "id": "sample5", 
                "type": "text",
                "title": "Ocean Beach",
                "content": "Beautiful ocean beach with waves and sand blue water sunset"
            }
        ]
        
        for item in sample_data:
            embedding = self.create_simple_embedding(item['content'])
            vector_id = f"text_{item['id']}"
            self.vectors[vector_id] = embedding
            self.metadata[vector_id] = {
                "type": "text",
                "title": item['title'],
                "content": item['content'],
                "image_data": None
            }
    
    def load_existing_images(self):
        """Load any existing images from uploads folder"""
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            print(f"🔍 Scanning {uploads_dir} folder for existing images...")
            for filename in os.listdir(uploads_dir):
                if allowed_file(filename):
                    file_path = os.path.join(uploads_dir, filename)
                    print(f"📸 Auto-loading: {filename}")
                    # Auto-analyze and add to database
                    try:
                        vector_id = self.add_image(file_path, "", filename)
                        print(f"✅ Loaded {vector_id}")
                    except Exception as e:
                        print(f"❌ Failed to load {filename}: {e}")
            print(f"🎉 Finished loading existing images!")

    def add_image(self, image_path: str, description: str, filename: str):
        """Add new image to database with better validation"""
        # Clean and validate description
        description = description.strip()
        if not description:
            print(f"Warning: No description provided for {filename}. Using filename only.")
            description = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('_', ' ')
        
        print(f"Adding image with description: '{description}'")
        
        # Create embedding primarily from description
        embedding = self.create_image_embedding(image_path, description)
        
        vector_id = f"image_{len([k for k in self.vectors.keys() if k.startswith('image_')])}"
        self.vectors[vector_id] = embedding
        
        image_base64 = self.image_to_base64(image_path)
        
        self.metadata[vector_id] = {
            "type": "image",
            "title": filename,
            "content": description,
            "image_data": image_base64,
            "path": image_path
        }
        
        print(f"Successfully added {vector_id} with embedding strength: {sum(x*x for x in embedding)**0.5:.3f}")
        return vector_id
    
    def search_similar(self, query: str, top_k: int = 5, filter_type: str = None):
        """Search for similar items with better debugging"""
        print(f"Searching for: '{query}' with filter: {filter_type}")
        
        query_embedding = self.create_simple_embedding(query)
        query_strength = sum(x*x for x in query_embedding)**0.5
        print(f"Query embedding strength: {query_strength:.3f}")
        
        similarities = []
        for vector_id, vector in self.vectors.items():
            metadata = self.metadata[vector_id]
            
            if filter_type and metadata['type'] != filter_type:
                continue
            
            similarity = self.cosine_similarity(query_embedding, vector)
            
            # Debug info
            if metadata['type'] == 'image':
                print(f"  {vector_id}: '{metadata['content']}' -> similarity: {similarity:.3f}")
            
            similarities.append({
                'id': vector_id,
                'score': similarity,
                'metadata': metadata
            })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"Top result: {similarities[0]['metadata']['content'] if similarities else 'None'}")
        return similarities[:top_k]
    
    def get_stats(self):
        """Get database statistics"""
        total = len(self.vectors)
        images = len([k for k in self.vectors.keys() if k.startswith('image_')])
        texts = len([k for k in self.vectors.keys() if k.startswith('text_')])
        
        return {
            'total': total,
            'images': images, 
            'texts': texts
        }

# Initialize database
db = ImageVectorDB()

@app.route('/')
def index():
    return render_template('image_search.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    description = request.form.get('description', '').strip()
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Add to database (auto-analysis will happen if no description)
        vector_id = db.add_image(file_path, description, filename)
        
        if not description:
            return jsonify({
                'success': True, 
                'message': f'Image uploaded and auto-analyzed as {vector_id}',
                'filename': filename,
                'auto_analyzed': True
            })
        else:
            return jsonify({
                'success': True, 
                'message': f'Image uploaded and indexed as {vector_id}',
                'filename': filename,
                'auto_analyzed': False
            })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint"""
    data = request.get_json()
    query = data.get('query', '')
    filter_type = data.get('filter', None)
    top_k = data.get('top_k', 5)
    
    if filter_type == 'all':
        filter_type = None
    
    results = db.search_similar(query, top_k, filter_type)
    return jsonify(results)

@app.route('/stats')
def get_stats():
    """Get database stats"""
    return jsonify(db.get_stats())

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("🚀 Starting Image Vector Database Web Interface...")
    print("🌐 Open your browser and go to: http://localhost:5001")
    print("📸 Upload images and search by description!")
    app.run(debug=True, host='0.0.0.0', port=5001)
