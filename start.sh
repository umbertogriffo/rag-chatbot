#!/bin/bash

# Start backend in background
cd backend
PYTHONPATH=.:../chatbot uvicorn main:app --reload &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
  sleep 1
done

echo "Backend is ready. Starting frontend..."

# Start frontend
cd ../frontend
yarn dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
