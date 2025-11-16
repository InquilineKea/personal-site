#!/usr/bin/env python3
"""
THERML Simple - Thermal Simulation Backend
A lightweight Python backend for thermal analysis and simulation
"""

import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import sys


class ThermalSimulator:
    """Core thermal simulation engine"""

    def __init__(self, grid_size=50, ambient_temp=20.0):
        self.grid_size = grid_size
        self.ambient_temp = ambient_temp
        self.grid = np.full((grid_size, grid_size), ambient_temp, dtype=float)
        self.diffusion_rate = 0.2
        self.cooling_rate = 0.01

    def add_heat_source(self, x, y, temperature, radius=3):
        """Add a heat source at specified coordinates"""
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))

        for i in range(max(0, x - radius), min(self.grid_size, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.grid_size, y + radius + 1)):
                distance = np.sqrt((i - x)**2 + (j - y)**2)
                if distance <= radius:
                    falloff = 1.0 - (distance / radius)
                    self.grid[i, j] = min(100.0, self.grid[i, j] + temperature * falloff)

    def step(self):
        """Perform one simulation step"""
        new_grid = self.grid.copy()

        # Heat diffusion using 2D convolution
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                neighbors_avg = (
                    self.grid[i-1, j] + self.grid[i+1, j] +
                    self.grid[i, j-1] + self.grid[i, j+1]
                ) / 4.0

                # Apply diffusion
                diffusion = (neighbors_avg - self.grid[i, j]) * self.diffusion_rate

                # Apply cooling toward ambient
                cooling = (self.grid[i, j] - self.ambient_temp) * self.cooling_rate

                new_grid[i, j] = self.grid[i, j] + diffusion - cooling
                new_grid[i, j] = np.clip(new_grid[i, j], 0, 100)

        self.grid = new_grid

    def get_statistics(self):
        """Calculate thermal statistics"""
        return {
            'min_temp': float(np.min(self.grid)),
            'max_temp': float(np.max(self.grid)),
            'avg_temp': float(np.mean(self.grid)),
            'std_temp': float(np.std(self.grid)),
            'hot_spots': int(np.sum(self.grid > 80)),
            'cold_spots': int(np.sum(self.grid < 30))
        }

    def get_grid_data(self):
        """Return grid as list for JSON serialization"""
        return self.grid.tolist()

    def reset(self):
        """Reset grid to ambient temperature"""
        self.grid = np.full((self.grid_size, self.grid_size), self.ambient_temp, dtype=float)

    def set_parameters(self, diffusion_rate=None, cooling_rate=None):
        """Update simulation parameters"""
        if diffusion_rate is not None:
            self.diffusion_rate = float(diffusion_rate)
        if cooling_rate is not None:
            self.cooling_rate = float(cooling_rate)


class ThermalHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for thermal simulation API"""

    simulator = ThermalSimulator()

    def _send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html_response(self, html, status=200):
        """Send HTML response"""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)

        if parsed.path == '/':
            self._send_html_response(self._get_index_html())

        elif parsed.path == '/api/status':
            self._send_json_response({'status': 'running', 'grid_size': self.simulator.grid_size})

        elif parsed.path == '/api/grid':
            data = {
                'grid': self.simulator.get_grid_data(),
                'stats': self.simulator.get_statistics()
            }
            self._send_json_response(data)

        elif parsed.path == '/api/stats':
            self._send_json_response(self.simulator.get_statistics())

        else:
            self._send_json_response({'error': 'Not found'}, 404)

    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json_response({'error': 'Invalid JSON'}, 400)
            return

        parsed = urlparse(self.path)

        if parsed.path == '/api/heat':
            x = data.get('x', 25)
            y = data.get('y', 25)
            temp = data.get('temperature', 75)
            radius = data.get('radius', 3)

            self.simulator.add_heat_source(x, y, temp, radius)
            self._send_json_response({'success': True, 'message': 'Heat source added'})

        elif parsed.path == '/api/step':
            steps = data.get('steps', 1)
            for _ in range(steps):
                self.simulator.step()

            self._send_json_response({
                'success': True,
                'stats': self.simulator.get_statistics()
            })

        elif parsed.path == '/api/reset':
            self.simulator.reset()
            self._send_json_response({'success': True, 'message': 'Simulation reset'})

        elif parsed.path == '/api/parameters':
            diffusion = data.get('diffusion_rate')
            cooling = data.get('cooling_rate')
            self.simulator.set_parameters(diffusion, cooling)
            self._send_json_response({'success': True, 'message': 'Parameters updated'})

        else:
            self._send_json_response({'error': 'Not found'}, 404)

    def do_OPTIONS(self):
        """Handle OPTIONS for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Custom log format"""
        sys.stdout.write(f"[THERML] {self.address_string()} - {format % args}\n")

    def _get_index_html(self):
        """Return simple HTML interface"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>THERML Python Backend</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #4CAF50; }
        .endpoint {
            background: #16213e;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        code {
            background: #0f3460;
            padding: 2px 6px;
            border-radius: 3px;
            color: #00d9ff;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover { background: #45a049; }
        #output {
            background: #0f3460;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>🔥 THERML Python Backend</h1>
    <p>Lightweight thermal simulation API server</p>

    <h2>API Endpoints:</h2>

    <div class="endpoint">
        <strong>GET /api/status</strong><br>
        Get server status and configuration
    </div>

    <div class="endpoint">
        <strong>GET /api/grid</strong><br>
        Get current thermal grid data and statistics
    </div>

    <div class="endpoint">
        <strong>GET /api/stats</strong><br>
        Get thermal statistics only
    </div>

    <div class="endpoint">
        <strong>POST /api/heat</strong><br>
        Add heat source: <code>{"x": 25, "y": 25, "temperature": 75, "radius": 3}</code>
    </div>

    <div class="endpoint">
        <strong>POST /api/step</strong><br>
        Run simulation steps: <code>{"steps": 10}</code>
    </div>

    <div class="endpoint">
        <strong>POST /api/reset</strong><br>
        Reset simulation to ambient temperature
    </div>

    <div class="endpoint">
        <strong>POST /api/parameters</strong><br>
        Update parameters: <code>{"diffusion_rate": 0.2, "cooling_rate": 0.01}</code>
    </div>

    <h2>Quick Test:</h2>
    <button onclick="testAPI('/api/status')">Test Status</button>
    <button onclick="testAPI('/api/stats')">Test Stats</button>
    <button onclick="testAddHeat()">Add Heat</button>
    <button onclick="testStep()">Run Step</button>
    <button onclick="testReset()">Reset</button>

    <div id="output"></div>

    <script>
        async function testAPI(endpoint) {
            const output = document.getElementById('output');
            try {
                const response = await fetch(endpoint);
                const data = await response.json();
                output.textContent = `Response from ${endpoint}:\\n` + JSON.stringify(data, null, 2);
            } catch (error) {
                output.textContent = `Error: ${error.message}`;
            }
        }

        async function testAddHeat() {
            const output = document.getElementById('output');
            try {
                const response = await fetch('/api/heat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x: 25, y: 25, temperature: 80, radius: 3})
                });
                const data = await response.json();
                output.textContent = 'Heat source added:\\n' + JSON.stringify(data, null, 2);
            } catch (error) {
                output.textContent = `Error: ${error.message}`;
            }
        }

        async function testStep() {
            const output = document.getElementById('output');
            try {
                const response = await fetch('/api/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({steps: 5})
                });
                const data = await response.json();
                output.textContent = 'Simulation stepped:\\n' + JSON.stringify(data, null, 2);
            } catch (error) {
                output.textContent = `Error: ${error.message}`;
            }
        }

        async function testReset() {
            const output = document.getElementById('output');
            try {
                const response = await fetch('/api/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: '{}'
                });
                const data = await response.json();
                output.textContent = 'Simulation reset:\\n' + JSON.stringify(data, null, 2);
            } catch (error) {
                output.textContent = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>
"""


def main():
    """Start the THERML backend server"""
    port = 8080

    print("=" * 60)
    print("🔥 THERML - Thermal Simulation Backend")
    print("=" * 60)
    print(f"Starting server on http://localhost:{port}")
    print(f"Grid size: {ThermalHTTPHandler.simulator.grid_size}x{ThermalHTTPHandler.simulator.grid_size}")
    print(f"Ambient temperature: {ThermalHTTPHandler.simulator.ambient_temp}°C")
    print("\nAPI Endpoints:")
    print("  GET  /api/status  - Server status")
    print("  GET  /api/grid    - Get thermal grid")
    print("  GET  /api/stats   - Get statistics")
    print("  POST /api/heat    - Add heat source")
    print("  POST /api/step    - Run simulation step")
    print("  POST /api/reset   - Reset simulation")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)

    try:
        server = HTTPServer(('0.0.0.0', port), ThermalHTTPHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()
        print("Server stopped.")


if __name__ == '__main__':
    main()
