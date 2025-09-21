# 🔥 Quick Heroku Deployment

## 1. Install Heroku CLI
Download from: https://devcenter.heroku.com/articles/heroku-cli

## 2. Login to Heroku
```bash
heroku login
```

## 3. Create Heroku App
```bash
cd "C:/Users/user/Desktop/Pinecone"
heroku create pinecone-vector-demo
```

## 4. Deploy
```bash
git push heroku main
```

## 5. Open Your App
```bash
heroku open
```

That's it! Your app will be live at: https://pinecone-vector-demo.herokuapp.com

## Optional: Set Environment Variables
```bash
heroku config:set PINECONE_API_KEY=your_api_key_here
heroku config:set PINECONE_ENVIRONMENT=your_environment_here
```