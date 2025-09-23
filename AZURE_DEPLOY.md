# 🌊 Azure Deployment Guide

## 🎯 Deploy to Azure App Service (Free Tier)

### Step 1: Prerequisites
- Azure account (free): https://azure.microsoft.com/free/
- Your GitHub repository: https://github.com/nagesh7588/pinecone

### Step 2: Create Azure App Service

1. **Login to Azure Portal**: https://portal.azure.com
2. **Create Resource** → **Web App**
3. **Configure**:
   - **Subscription**: Free subscription
   - **Resource Group**: Create new `pinecone-rg`
   - **Name**: `pinecone-vector-demo` (must be unique)
   - **Runtime Stack**: `Python 3.12`
   - **Region**: Choose closest to you
   - **Pricing Plan**: `F1 Free` (100% free)

### Step 3: Deploy from GitHub

1. **In your App Service** → **Deployment Center**
2. **Source**: GitHub
3. **Organization**: nagesh7588
4. **Repository**: pinecone
5. **Branch**: main
6. **Click Save**

### Step 4: Configure Environment

1. **Configuration** → **Application Settings**
2. **Add** environment variables if needed:
   ```
   PINECONE_API_KEY = your_key_here
   PINECONE_ENVIRONMENT = your_env_here
   ```

### Step 5: Access Your App

Your app will be live at: `https://pinecone-vector-demo.azurewebsites.net`

## 🔄 Automatic Deployments

Every Git push to main branch automatically deploys to Azure!

```bash
git add .
git commit -m "Update app"
git push origin main
# ✅ Auto-deploys to Azure
```

## 💰 Azure Free Tier Limits

- **Compute**: 60 minutes/day
- **Storage**: 1GB
- **Bandwidth**: 165MB/day outbound
- **Custom Domain**: Available
- **SSL**: Included

## 🎉 Benefits

✅ **Free hosting**
✅ **GitHub integration** 
✅ **Auto-deployments**
✅ **Azure ecosystem**
✅ **Professional URL**
✅ **SSL certificate**

Perfect for development and small projects!