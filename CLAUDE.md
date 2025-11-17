# CLAUDE.md - AI Assistant Guide

This document provides comprehensive guidance for AI assistants working with this codebase.

## Project Overview

**Type**: Static HTML Personal Website
**Purpose**: Simple, minimal personal website showcasing cat enthusiast content
**Tech Stack**: Pure HTML5 + MVP.css framework
**Deployment**: Fly.io via Docker containerization
**Repository Size**: ~341 KB with 8 non-git files

## Codebase Structure

```
/home/user/personal-site/
├── index.html              # Main HTML file (68 lines) - the entire website
├── Dockerfile              # Container image definition (4 lines)
├── fly.toml                # Fly.io deployment configuration (24 lines)
├── README.md               # Project documentation
├── .dockerignore           # Docker build exclusions
└── img/                    # Image assets directory
    ├── 200.jpg             # 400x200 JPEG (9.3 KB)
    ├── 225.jpg             # 400x225 JPEG (17 KB)
    └── 250.jpg             # 400x250 JPEG (9.4 KB)
```

### Key Files

- **`index.html`**: The entire website in a single file. All content and structure lives here.
- **`Dockerfile`**: Builds a container using `pierrezemb/gostatic` Go-based static server
- **`fly.toml`**: Defines deployment settings for Fly.io platform
- **`img/`**: Contains all image assets in JPEG format

## Technology Stack

### Frontend
- **HTML5**: Semantic markup following web standards
- **MVP.css**: Minimal CSS framework loaded from CDN
  - URL: `https://unpkg.com/mvp.css`
  - No custom CSS files
  - Provides automatic styling via semantic HTML
  - Zero JavaScript required

### Backend/Infrastructure
- **Web Server**: `pierrezemb/gostatic` (Go-based static file server)
  - Serves files from `/srv/http/` in container
  - Runs on port 8080
  - HTTPS promotion enabled
  - Logging enabled

### Deployment
- **Platform**: Fly.io
  - App name: `personal-site-2zkxcg`
  - Primary region: `iad` (Northern Virginia)
  - Auto-scaling: min 0 machines (scales to demand)
  - Resources: 1GB RAM, 1 shared vCPU
  - Force HTTPS enabled

### Build Tools
**NONE** - This is a pure static site with zero build pipeline:
- No npm/Node.js
- No webpack/bundlers
- No preprocessors
- No package managers

## Development Workflows

### Local Development

1. **Direct File Opening** (simplest):
   ```bash
   # Open directly in browser
   open index.html  # macOS
   xdg-open index.html  # Linux
   ```

2. **Local HTTP Server** (recommended for accurate testing):
   ```bash
   # Python 3
   python -m http.server 8080

   # Python 2
   python -m SimpleHTTPServer 8080

   # Node.js (if available)
   npx http-server -p 8080

   # PHP
   php -S localhost:8080
   ```

3. **Docker Testing** (production-like):
   ```bash
   docker build -t personal-site .
   docker run -p 8080:8080 personal-site
   # Visit http://localhost:8080
   ```

### Deployment Workflow

```
Edit index.html → Commit → Push → Fly.io Auto-Deploy → Live Site
```

**Deployment Command**:
```bash
# Manual deployment (if needed)
fly deploy

# Check deployment status
fly status

# View logs
fly logs
```

### Git Workflow

**Current Branch**: `claude/claude-md-mi2j3amh5w73odnj-01QNs4GSs6Nu1eGCTHew1E7C`

**Standard Process**:
```bash
# Make changes
git add .
git commit -m "Description of changes"

# Push to current branch (ALWAYS use -u flag)
git push -u origin claude/claude-md-mi2j3amh5w73odnj-01QNs4GSs6Nu1eGCTHew1E7C
```

**Important Git Rules**:
- Always develop on designated `claude/` branches
- Branch names MUST start with `claude/` and match session ID
- Use descriptive commit messages
- Push with `-u origin <branch-name>` flag
- Retry network failures up to 4 times with exponential backoff (2s, 4s, 8s, 16s)

## Code Conventions and Patterns

### HTML Structure Patterns

1. **Semantic HTML5 Elements**:
   ```html
   <main>
     <article>...</article>
     <hr>
     <section>
       <header><h2>...</h2></header>
       <aside>...</aside>
     </section>
   </main>
   ```

2. **Content Cards** (repeating pattern):
   ```html
   <aside>
     <img alt="Descriptive text" src="img/filename.jpg" height="150">
     <h3><a href="https://social-link">Name</a></h3>
     <blockquote>
       "Quote content"
       <footer><i>Attribution</i></footer>
     </blockquote>
   </aside>
   ```

3. **Required Meta Tags**:
   ```html
   <meta charset="utf-8">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   <link rel="stylesheet" href="https://unpkg.com/mvp.css">
   ```

### Styling Guidelines

- **NO custom CSS**: Rely entirely on MVP.css semantic styling
- **NO inline styles**: Keep HTML clean and semantic
- **NO CSS classes**: MVP.css works via HTML element selectors
- **Component organization**: Use semantic elements for automatic styling

### Accessibility Standards

✅ **MUST Follow**:
- Always include `alt` attributes on images
- Maintain proper heading hierarchy (h1 → h2 → h3)
- Use semantic HTML elements (`<article>`, `<aside>`, `<section>`, etc.)
- Include viewport meta tag for responsive design
- Ensure sufficient color contrast (handled by MVP.css)

### Image Guidelines

- **Format**: JPEG for photos
- **Naming**: Descriptive names (e.g., `200.jpg` for 200px height)
- **Location**: All images in `/img/` directory
- **Optimization**: Keep file sizes small (under 20 KB per image)
- **Dimensions**: Use reasonable resolutions (400px width is typical)
- **Display**: Set `height` attribute (e.g., `height="150"`) for consistency

## Common Tasks for AI Assistants

### Adding New Content

1. **Add new section**:
   - Insert after existing `<hr>` divider
   - Use `<section>` wrapper with `<header>` and content
   - Follow existing semantic structure

2. **Add new card/profile**:
   - Add new `<aside>` element within section
   - Include image, heading link, and blockquote
   - Maintain consistent structure

3. **Update text content**:
   - Edit directly in `index.html`
   - Preserve HTML entity encoding (e.g., `&` → `&amp;`)
   - Keep semantic markup intact

### Modifying Configuration

1. **Update Fly.io settings** → Edit `fly.toml`
   - Memory/CPU allocation in `[[vm]]` section
   - Port settings in `[http_service]`
   - Auto-scaling parameters

2. **Change Docker configuration** → Edit `Dockerfile`
   - Server flags in `CMD` directive
   - Port changes (remember to update `fly.toml` too)

### Adding Images

1. Place image in `/img/` directory
2. Use descriptive filename
3. Reference in HTML: `<img src="img/filename.jpg" alt="Description" height="150">`
4. Optimize before adding (keep under 20 KB if possible)

## Testing Checklist

Since there's no automated testing, manually verify:

- [ ] HTML validates (use W3C validator)
- [ ] All images load correctly
- [ ] Links work and point to correct destinations
- [ ] Responsive design works on mobile (test viewport)
- [ ] Semantic HTML structure is maintained
- [ ] Alt text exists for all images
- [ ] Docker container builds successfully
- [ ] Local server runs without errors
- [ ] Content displays correctly across browsers

## Deployment Checklist

Before pushing to production:

- [ ] Test locally with Docker: `docker build -t personal-site . && docker run -p 8080:8080 personal-site`
- [ ] Verify all links are correct (no broken URLs)
- [ ] Check image paths are relative (start with `img/`)
- [ ] Ensure no absolute localhost URLs
- [ ] Git commit with descriptive message
- [ ] Push to correct `claude/` branch with `-u` flag
- [ ] Monitor Fly.io deployment logs if deploying manually
- [ ] Verify live site after deployment

## Important Constraints

### What NOT to Do

❌ **DO NOT**:
- Add npm, package.json, or Node.js dependencies
- Create custom CSS files (use MVP.css semantic styling)
- Add JavaScript unless absolutely necessary (goes against project simplicity)
- Use CSS frameworks other than MVP.css
- Add build tools or preprocessors
- Create complex directory structures
- Use CSS classes or IDs for styling
- Add testing frameworks (no test infrastructure exists)

### What TO Do

✅ **DO**:
- Keep it simple - maintain the minimalist philosophy
- Use semantic HTML for all structure
- Follow existing patterns when adding content
- Preserve accessibility features
- Test in Docker before deploying
- Keep images optimized and small
- Use descriptive commit messages
- Maintain backward compatibility

## Security Considerations

- **No user input**: Static site has no forms or dynamic content
- **HTTPS enforced**: Fly.io configuration forces HTTPS
- **No secrets**: No API keys, passwords, or sensitive data
- **CDN dependency**: MVP.css loaded from unpkg.com (reliable CDN)
- **Docker security**: Using minimal Go-based static server (small attack surface)

## Troubleshooting

### Common Issues

**Issue**: Docker build fails
- **Solution**: Check Dockerfile syntax, ensure all files exist

**Issue**: Fly.io deployment fails
- **Solution**: Verify `fly.toml` is valid, check app name matches

**Issue**: Images not loading locally
- **Solution**: Use HTTP server instead of `file://` protocol

**Issue**: CSS not loading
- **Solution**: Check internet connection (MVP.css is CDN-only)

**Issue**: Git push fails with 403
- **Solution**: Verify branch name starts with `claude/` and matches session ID

## External Resources

- **MVP.css Documentation**: https://andybrewer.github.io/mvp/
- **Fly.io Documentation**: https://fly.io/docs/
- **Original Template**: https://github.com/andybrewer/mvp/
- **Live Demo**: https://radekosmulski.github.io/personal-site/

## Project Philosophy

This project embraces **radical simplicity**:
- No JavaScript
- No build tools
- No package managers
- No custom CSS
- Single HTML file
- Minimal dependencies
- Fast loading
- Easy to understand
- Easy to modify

**When making changes, always ask**: "Does this maintain the simplicity?"

## File References

Key locations for common modifications:

- **Page title**: `/home/user/personal-site/index.html:5`
- **Main heading**: `/home/user/personal-site/index.html:9-10`
- **Content sections**: `/home/user/personal-site/index.html:15-50`
- **Footer**: `/home/user/personal-site/index.html:64-66`
- **Docker port**: `/home/user/personal-site/Dockerfile:3`
- **Fly.io port**: `/home/user/personal-site/fly.toml:4`
- **Fly.io resources**: `/home/user/personal-site/fly.toml:13-16`

---

**Last Updated**: 2025-11-17
**Repository**: InquilineKea/personal-site
**Purpose**: Guide for AI assistants working with this minimal static site
