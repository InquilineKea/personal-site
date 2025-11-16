# Deployment Guide

This project is ready to deploy to Railway, Fly.io, or any platform that supports Python web apps.

## Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/InquilineKea/personal-site)

### Option 1: Railway (Recommended - Easiest)

1. Go to [Railway.app](https://railway.app/)
2. Sign up/login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select this repository
5. Railway will auto-detect the Python app and deploy it
6. Your app will be live at a `railway.app` URL!

### Option 2: Fly.io

1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Login: `flyctl auth login`
3. Create app: `flyctl launch` (follow prompts)
4. Deploy: `flyctl deploy`

### Option 3: Render

1. Go to [Render.com](https://render.com/)
2. New → Web Service
3. Connect your GitHub repo
4. Render auto-detects Python
5. Deploy!

### Option 4: Heroku

```bash
heroku create your-app-name
git push heroku main
```

## Files Included for Deployment

- `server.py` - Flask web server
- `requirements.txt` - Python dependencies
- `Procfile` - Process configuration
- `railway.json` - Railway-specific config

## Local Testing

```bash
pip install -r requirements.txt
python server.py
# Visit http://localhost:8080
```

## What Gets Deployed

All your THRML implementations:
- `index.html` - Main landing page
- `therml_simple.py` - Python Ising model
- `therml_demo.html` - Simple interactive demo
- `therml_full.html` - Complete THRML implementation

## Environment Variables

The server automatically uses the `PORT` environment variable provided by the hosting platform. No configuration needed!
