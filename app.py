from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = 5000
    print("=" * 80)
    print("Flask app is running on port", port)
    print("Starting tunnel service...")
    print("=" * 80)
    app.run(host='127.0.0.1', port=port, debug=False)
