#!/bin/bash

# Start backend in background
cd backend || exit
PYTHONPATH=.:../chatbot poetry run python migration.py
PYTHONPATH=.:../chatbot uvicorn main:app &
BACKEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT

# Wait for backend to be ready
echo "Waiting for backend to start..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
  sleep 1
done

echo "Backend is ready. Starting frontend..."

# Start frontend
cd ../frontend || exit
yarn dev
