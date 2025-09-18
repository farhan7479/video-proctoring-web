#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs/sessions logs/reports logs/events

# Start the server
uvicorn main:app --host 0.0.0.0 --port $PORT