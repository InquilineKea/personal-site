# Deployment Guide

This guide provides step-by-step instructions for deploying your personal site to Fly.io and Railway, as well as running it locally with Docker.

## Table of Contents

- [Local Development with Docker](#local-development-with-docker)
- [Deploying to Fly.io](#deploying-to-flyio)
- [Deploying to Railway](#deploying-to-railway)
- [Custom Domains](#custom-domains)
- [Troubleshooting](#troubleshooting)

---

## Local Development with Docker

### Prerequisites

- Docker installed ([Download Docker](https://www.docker.com/products/docker-desktop))
- Docker Compose installed (included with Docker Desktop)

### Running Locally

1. **Build and start the container:**

   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the Docker image
   - Start the Flask app on port 8080
   - Enable hot-reloading for development

2. **Access your site:**

   Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

3. **Stop the container:**

   Press `Ctrl+C` in the terminal, or run:
   ```bash
   docker-compose down
   ```

### Useful Docker Commands

```bash
# Rebuild the image (if you change dependencies)
docker-compose build

# Run in detached mode (background)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop and remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v
```

---

## Deploying to Fly.io

Fly.io is a platform for running applications close to your users globally.

### Prerequisites

- A Fly.io account ([Sign up here](https://fly.io/app/sign-up))
- flyctl CLI installed ([Installation guide](https://fly.io/docs/hands-on/install-flyctl/))

### Installation (macOS/Linux)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Add to PATH (add to your .bashrc or .zshrc)
export FLYCTL_INSTALL="/home/$USER/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
```

### Installation (Windows)

```powershell
# Using PowerShell
iwr https://fly.io/install.ps1 -useb | iex
```

### Deployment Steps

1. **Authenticate with Fly.io:**

   ```bash
   flyctl auth login
   ```

   This will open a browser window for authentication.

2. **Update the app name in fly.toml:**

   Open `fly.toml` and change the app name (line 5):
   ```toml
   app = "your-unique-app-name"
   ```

   The app name must be globally unique across all Fly.io apps.

3. **Launch your app:**

   ```bash
   flyctl launch --no-deploy
   ```

   This will:
   - Use the existing `fly.toml` configuration
   - Create the app in Fly.io
   - NOT deploy yet (we'll do this manually)

   When prompted:
   - Choose your preferred region (or use default: `iad` - Ashburn, VA)
   - Decline setting up PostgreSQL (not needed for static site)
   - Decline setting up Redis (not needed)

4. **Deploy your application:**

   ```bash
   flyctl deploy
   ```

   This will:
   - Build your Docker image
   - Push it to Fly.io's registry
   - Deploy the application
   - Provide you with a URL (e.g., `https://your-app-name.fly.dev`)

5. **View your deployed app:**

   ```bash
   flyctl open
   ```

   Or visit: `https://your-app-name.fly.dev`

### Fly.io Management Commands

```bash
# Check app status
flyctl status

# View logs
flyctl logs

# Check health and deployments
flyctl checks list

# Scale your app
flyctl scale count 2

# SSH into your app
flyctl ssh console

# View app secrets (environment variables)
flyctl secrets list

# Set a secret
flyctl secrets set MY_SECRET=value

# Restart your app
flyctl apps restart

# Destroy your app (careful!)
flyctl apps destroy your-app-name
```

### Updating Your App

After making changes to your code:

```bash
flyctl deploy
```

That's it! Fly.io will build and deploy the new version.

### Cost Optimization

The current configuration (`fly.toml`) includes:
- `auto_stop_machines = true` - Stops machines when idle
- `auto_start_machines = true` - Starts machines on request
- `min_machines_running = 0` - Allows all machines to stop
- 256 MB RAM, 1 shared CPU

This configuration stays within Fly.io's free tier for hobby projects.

---

## Deploying to Railway

Railway is a modern platform that makes deployment simple with automatic HTTPS and custom domains.

### Prerequisites

- A Railway account ([Sign up here](https://railway.app/))
- Railway CLI (optional but recommended) ([Installation guide](https://docs.railway.app/develop/cli))

### Method 1: Deploy via GitHub (Recommended)

1. **Push your code to GitHub:**

   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Create a new project on Railway:**

   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your GitHub account
   - Select your repository

3. **Configure the deployment:**

   Railway will automatically detect the Dockerfile and deploy. No additional configuration needed!

4. **Wait for deployment:**

   Railway will:
   - Build your Docker image
   - Deploy the application
   - Provide a public URL

5. **Access your app:**

   Click "Generate Domain" in the Railway dashboard to get a public URL like:
   ```
   https://your-app-name.up.railway.app
   ```

### Method 2: Deploy via CLI

1. **Install Railway CLI:**

   ```bash
   # macOS/Linux
   curl -fsSL https://railway.app/install.sh | sh

   # Windows (with Scoop)
   scoop install railway

   # NPM (all platforms)
   npm i -g @railway/cli
   ```

2. **Login to Railway:**

   ```bash
   railway login
   ```

3. **Initialize your project:**

   ```bash
   railway init
   ```

   Follow the prompts to create a new project or link to an existing one.

4. **Deploy:**

   ```bash
   railway up
   ```

5. **Generate a domain:**

   ```bash
   railway domain
   ```

### Railway Management Commands

```bash
# View logs
railway logs

# Check status
railway status

# Link to a project
railway link

# Open dashboard
railway open

# Set environment variables
railway variables set KEY=value

# List environment variables
railway variables

# Delete a variable
railway variables delete KEY

# Restart the service
railway restart
```

### Environment Variables (Optional)

If you need to add environment variables:

1. **Via Dashboard:**
   - Go to your project in Railway
   - Click on "Variables"
   - Add your key-value pairs

2. **Via CLI:**
   ```bash
   railway variables set PORT=8080
   railway variables set FLASK_ENV=production
   ```

### Updating Your App

With GitHub integration:
```bash
git add .
git commit -m "Update site"
git push origin main
```

Railway will automatically detect the push and redeploy.

With CLI:
```bash
railway up
```

---

## Custom Domains

### Fly.io Custom Domains

1. **Add your domain:**

   ```bash
   flyctl certs add yourdomain.com
   flyctl certs add www.yourdomain.com
   ```

2. **Update DNS records:**

   Add the following DNS records at your domain registrar:

   ```
   Type: A
   Name: @
   Value: [IP from flyctl certs show yourdomain.com]

   Type: AAAA
   Name: @
   Value: [IPv6 from flyctl certs show yourdomain.com]

   Type: CNAME
   Name: www
   Value: yourdomain.com
   ```

3. **Verify SSL certificate:**

   ```bash
   flyctl certs show yourdomain.com
   ```

   Wait for the certificate status to show "Ready".

### Railway Custom Domains

1. **Add domain in Dashboard:**
   - Go to your Railway project
   - Click "Settings" → "Domains"
   - Click "Add Custom Domain"
   - Enter your domain

2. **Update DNS records:**

   Add a CNAME record at your domain registrar:

   ```
   Type: CNAME
   Name: @ (or www)
   Value: [CNAME target provided by Railway]
   ```

3. **Wait for verification:**

   Railway will automatically provision an SSL certificate via Let's Encrypt.

---

## Troubleshooting

### Docker Issues

**Problem:** Container won't start
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

**Problem:** Port already in use
```bash
# Find process using port 8080
lsof -i :8080  # macOS/Linux
netstat -ano | findstr :8080  # Windows

# Kill the process or change port in docker-compose.yml
```

### Fly.io Issues

**Problem:** Deployment fails
```bash
# Check build logs
flyctl logs

# Try local build
docker build -t personal-site .
docker run -p 8080:8080 personal-site
```

**Problem:** App is unreachable
```bash
# Check app status
flyctl status

# Check health checks
flyctl checks list

# Restart app
flyctl apps restart
```

**Problem:** App name already taken
- Edit `fly.toml` and change the `app` name to something unique
- Run `flyctl launch --no-deploy` again

### Railway Issues

**Problem:** Build fails
- Check the build logs in Railway dashboard
- Ensure Dockerfile is in the root directory
- Verify all required files are committed to git

**Problem:** App crashes on startup
- Check logs in Railway dashboard
- Verify PORT environment variable is set correctly
- Ensure gunicorn is in requirements.txt

### General Issues

**Problem:** Static files not loading
- Verify the `img` directory is present
- Check that files aren't listed in `.dockerignore`
- Ensure Flask is serving static files correctly

**Problem:** Health check failing
```bash
# Test locally
curl http://localhost:8080/health

# Check if gunicorn is running
ps aux | grep gunicorn
```

---

## Additional Resources

### Fly.io
- [Official Documentation](https://fly.io/docs/)
- [Python on Fly.io](https://fly.io/docs/languages-and-frameworks/python/)
- [Pricing](https://fly.io/docs/about/pricing/)

### Railway
- [Official Documentation](https://docs.railway.app/)
- [Deploying with Docker](https://docs.railway.app/deploy/dockerfiles)
- [Pricing](https://railway.app/pricing)

### Docker
- [Official Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## Quick Reference

### Local Development
```bash
docker-compose up --build    # Start
docker-compose down          # Stop
```

### Fly.io
```bash
flyctl auth login            # Login
flyctl launch --no-deploy    # Create app
flyctl deploy                # Deploy
flyctl logs                  # View logs
flyctl open                  # Open in browser
```

### Railway
```bash
railway login                # Login
railway init                 # Create project
railway up                   # Deploy
railway logs                 # View logs
railway open                 # Open dashboard
```

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the platform-specific documentation
3. Check application logs
4. Verify all configuration files are correct
5. Ensure dependencies are up to date

For platform-specific issues:
- Fly.io: https://community.fly.io/
- Railway: https://discord.gg/railway

---

Happy deploying! 🚀
