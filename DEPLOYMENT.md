# Flask Deployment with Public URL

This repository contains a Flask app that serves your personal website with free public URL access using localtunnel.

## Quick Start

### Option 1: Using the startup script (Recommended)
```bash
chmod +x start_server.sh
./start_server.sh
```

### Option 2: Using Node.js tunnel script
```bash
# Install dependencies
pip install Flask==3.0.0
npm install localtunnel

# Terminal 1: Start Flask
python app.py

# Terminal 2: Start tunnel
node tunnel.js
```

The tunnel URL will be displayed in the console and saved to `/tmp/tunnel_url.txt`.

### Option 3: Manual setup
```bash
# Install dependencies
pip install Flask==3.0.0
npm install -g localtunnel

# Start Flask (in one terminal)
python app.py

# Start tunnel (in another terminal)
lt --port 5000
```

## Alternative Free Tunneling Services

If localtunnel doesn't work in your environment, try these alternatives:

### ngrok (requires signup but has free tier)
```bash
# Install ngrok from https://ngrok.com/download
# After installing and setting up auth token:
python app.py &
ngrok http 5000
```

### Cloudflare Tunnel (free, no signup needed for quick tunnels)
```bash
# Install cloudflared
# Run quick tunnel:
python app.py &
cloudflared tunnel --url http://localhost:5000
```

### serveo.net (SSH-based, no installation)
```bash
python app.py &
ssh -R 80:localhost:5000 serveo.net
```

### localhost.run (SSH-based, no installation)
```bash
python app.py &
ssh -R 80:localhost:5000 localhost.run
```

## Files

- `app.py` - Flask application server
- `tunnel.js` - Node.js script for creating public tunnel
- `start_server.sh` - All-in-one startup script
- `requirements.txt` - Python dependencies
- `package.json` - Node.js dependencies

## Network Restrictions

If you're in an environment with network restrictions (corporate proxy, firewall, etc.):

1. The Flask app will still work locally at `http://localhost:5000`
2. Tunneling services may be blocked - try running this on your local machine or a cloud VM
3. Consider deploying to a free hosting service instead (see below)

## Free Hosting Alternatives

Instead of using tunneling services, consider these free hosting options:

- **GitHub Pages** (static sites only - already set up for this repo)
- **Vercel** (supports Flask via serverless functions)
- **Railway** (free tier available)
- **Render** (free tier available)
- **PythonAnywhere** (free tier available)
- **Fly.io** (free tier available)

## Troubleshooting

### Flask not starting
- Check if port 5000 is already in use: `lsof -i :5000`
- Try a different port by editing `app.py`

### Tunnel not connecting
- Check your internet connection
- Try alternative tunneling services listed above
- Check if your firewall/proxy is blocking the service

### Images not loading
- Make sure the `img/` directory exists with the required images
- Check browser console for errors
