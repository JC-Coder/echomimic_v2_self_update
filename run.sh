#!/bin/bash

# Start HTTP server in background
python3 -m http.server 8005 &

# Start API server
python3.10 api_server.py 
