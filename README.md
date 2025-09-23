# 🌲 Pinecone Vector Database Demo

A simple Python Flask application demonstrating vector database capabilities for document search.

## 🚀 Features

- **Document Search**: Semantic search through programming documents
- **Web Interface**: Clean Flask-based web UI
- **Local Embeddings**: Works without external APIs
- **Azure Ready**: Optimized for Azure App Service deployment

## 📁 Project Structure

```
pinecone/
├── main.py              # CLI demo with Pinecone cloud
├── app.py              # Main Flask web application
├── application.py      # Azure App Service entry point
├── requirements.txt    # Python dependencies
├── web.config         # Azure configuration
├── templates/         # HTML templates
│   └── index.html    # Document search interface
└── README.md         # This file
```

## 🛠️ Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nagesh7588/pinecone.git
   cd pinecone
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally**:
   ```bash
   python app.py
   ```
   Open browser to `http://localhost:5000`

## 🌊 Azure Deployment (Free Tier)

See [AZURE_DEPLOY.md](AZURE_DEPLOY.md) for complete deployment guide.

**Quick Steps**:
1. Create Azure App Service (F1 Free tier)
2. Connect to GitHub repository
3. Deploy automatically

**Live URL**: `https://your-app-name.azurewebsites.net`

## 🎯 Sample Searches

- **"python"** → Python Programming + Data Science docs
- **"web development"** → HTML, CSS, JavaScript frameworks
- **"machine learning"** → AI, neural networks, algorithms
- **"design"** → UX design, usability, prototyping

## 🧠 How It Works

1. **Text Processing**: Converts documents into numerical vectors
2. **Similarity Search**: Uses cosine similarity to find matches
3. **Ranking**: Results sorted by relevance score

## 📦 Dependencies

- **Flask**: Web framework
- **Pinecone**: Vector database (optional, for cloud features)
- **python-dotenv**: Environment variable management

##  Links

- **GitHub Repository**: https://github.com/nagesh7588/pinecone
- **Azure Documentation**: https://docs.microsoft.com/azure/app-service/
- **Flask Documentation**: https://flask.palletsprojects.com/

---

**Deploy to Azure in minutes!** 🌊
