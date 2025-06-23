#!/bin/bash

# Function to stop servers
stop_servers() {
    if [ -f "http_server.pid" ]; then
        echo "Stopping HTTP server..."
        kill $(cat http_server.pid)
        rm http_server.pid
    fi
}

# Handle Ctrl+C
trap 'stop_servers; exit' INT

# Stop any existing servers
stop_servers

# Start HTTP server in background and save PID
python3 -m http.server 8005 &
echo $! > http_server.pid

echo "Servers started!"
echo "HTTP server running on http://localhost:8005"
echo "API server running on http://localhost:8000"
echo "Access the interface at: http://localhost:8005/api_client.html"
echo "Press Ctrl+C to stop servers"

# Start API server (this will run in foreground)
python3.10 api_server.py 
