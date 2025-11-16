#!/bin/bash

echo "================================================================================"
echo "Starting Flask App with Public URL"
echo "================================================================================"

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install Flask==3.0.0
fi

# Check if localtunnel is installed
if ! command -v lt &> /dev/null; then
    echo "Installing localtunnel..."
    npm install -g localtunnel
fi

# Start Flask app in background
echo "Starting Flask app on port 5000..."
python app.py &
FLASK_PID=$!

# Wait for Flask to start
sleep 3

# Start localtunnel
echo "Creating public tunnel..."
lt --port 5000 &
TUNNEL_PID=$!

# Wait a moment for tunnel to connect
sleep 5

echo "================================================================================"
echo "Setup complete!"
echo "Flask PID: $FLASK_PID"
echo "Tunnel PID: $TUNNEL_PID"
echo ""
echo "To stop the servers, run:"
echo "  kill $FLASK_PID $TUNNEL_PID"
echo "================================================================================"

# Keep script running
wait
