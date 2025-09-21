# 🌲 Pinecone Vector Database Demo

A comprehensive Python project demonstrating vector database capabilities using Pinecone with both structured and unstructured data handling.

## 🚀 Features

- **Local Vector Database**: Complete local implementation without external APIs
- **Document Search**: Semantic search through programming documents
- **Image Support**: Upload and search images with auto-analysis
- **Web Interface**: Beautiful Flask-based web UI
- **No External Dependencies**: Works completely offline with simple embeddings

## 📁 Project Structure

```
pinecone/
├── main.py                 # Main Pinecone demo with CLI interface
├── web_app.py             # Web interface for document search
├── image_web_app.py       # Web interface with image support
├── local_main.py          # Fully local vector database demo
├── image_vector_db.py     # Standalone image processing module
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates for web interface
│   ├── index.html        # Document search interface
│   └── image_search.html # Image search interface
└── README.md             # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nagesh7588/pinecone.git
   cd pinecone
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional, for Pinecone cloud):
   ```bash
   cp .env.example .env
   # Edit .env with your Pinecone API key and environment
   ```

## 🎮 Usage

### 1. Document Search Web Interface
```bash
python web_app.py
```
- Open browser to `http://localhost:5000`
- Search through programming documents
- Topics: Python, Web Development, Data Science, UX Design, Machine Learning

### 2. Image Search Web Interface
```bash
python image_web_app.py
```
- Open browser to `http://localhost:5001`
- Upload images and search by keywords
- Auto-analysis of image content (colors, textures, scenes)

### 3. Command Line Demo
```bash
python main.py
```
- Demonstrates Pinecone cloud integration
- Shows similarity search results in terminal

### 4. Local Vector Database Demo
```bash
python local_main.py
```
- Completely local implementation
- No external APIs required

## 🎯 Sample Searches

### Document Search:
- **"python"** → Python Programming + Data Science docs
- **"web development"** → HTML, CSS, JavaScript frameworks
- **"machine learning"** → AI, neural networks, algorithms
- **"design"** → UX design, usability, prototyping

### Image Search:
- Upload any image and search by:
  - Colors: "red", "blue", "bright"
  - Content: "cat", "nature", "building"
  - Scene: "outdoor", "indoor", "landscape"

## 🧠 How It Works

### Vector Embeddings
1. **Text Processing**: Converts documents into numerical vectors
2. **Similarity Search**: Uses cosine similarity to find matches
3. **Ranking**: Results sorted by relevance score

### Image Analysis
1. **Color Analysis**: Extracts dominant colors and brightness
2. **Texture Detection**: Analyzes image patterns and contrast
3. **Auto-Description**: Generates keywords from visual features
4. **Semantic Matching**: Combines visual and text features

## 🛡️ Environment Variables

Create a `.env` file for Pinecone cloud features:

```env
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=your_environment_here
```

## 📦 Dependencies

- **Flask**: Web framework for UI
- **Pinecone**: Vector database (cloud)
- **Pillow**: Image processing
- **NumPy**: Numerical computations
- **python-dotenv**: Environment variable management

## 🌐 Deployment

### GitHub Pages (Static)
1. Push to GitHub repository
2. Enable GitHub Pages in repository settings
3. Deploy static files (HTML/CSS/JS only)

### Heroku (Full Stack)
1. Create `Procfile`:
   ```
   web: python web_app.py
   ```
2. Deploy to Heroku with Python buildpack

### Vercel/Netlify (Serverless)
1. Configure for Python Flask deployment
2. Set environment variables in platform

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

- **GitHub Repository**: https://github.com/nagesh7588/pinecone
- **Pinecone Documentation**: https://docs.pinecone.io/
- **Flask Documentation**: https://flask.palletsprojects.com/

## 🎉 Features Showcase

- ✅ **Local Vector Database**: No external APIs needed
- ✅ **Image Search**: Upload and find images by description
- ✅ **Document Search**: Semantic search through tech documents
- ✅ **Web Interface**: Beautiful, responsive UI
- ✅ **Auto-Analysis**: AI-powered image content detection
- ✅ **Real-time Search**: Instant results as you type
- ✅ **Mobile Friendly**: Works on all devices

---

**Made with ❤️ using Python, Flask, and Pinecone**
