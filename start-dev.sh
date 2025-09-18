#!/bin/bash

echo "🚀 Starting AI Proctor Development Environment..."

# Kill existing processes
pkill -f "python3 main.py" 2>/dev/null
pkill -f "vite" 2>/dev/null

# Get absolute path
PROJECT_DIR=$(pwd)

# Start backend
echo "📡 Starting Backend (Python FastAPI)..."
cd $PROJECT_DIR/backend && $PROJECT_DIR/.venv/bin/python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend running at http://localhost:8000"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "🎨 Starting Frontend (Vite)..."
cd $PROJECT_DIR/frontend && npm run dev &
FRONTEND_PID=$!

echo "🎉 Development environment started!"
echo "📡 Backend: http://localhost:8000"
echo "🎨 Frontend: http://localhost:5175 (or check terminal for actual port)"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for Ctrl+C
trap 'echo "🛑 Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID; exit' INT
wait