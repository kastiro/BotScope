#!/bin/bash

# Server Runner (Linux/macOS)

echo "[INFO] Starting servers..."

if [ -f ".venv/bin/activate" ]; then
    VENV_ACTIVATE=".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    VENV_ACTIVATE="venv/bin/activate"
else
    echo "[ERROR] Virtual environment not found."
    exit 1
fi

source "$VENV_ACTIVATE"

# Run Backend in background
echo "[INFO] Starting Backend API (Port 8000)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Run BotScope in background
echo "[INFO] Starting BotScope (Port 4000)..."
uvicorn botdetect.main:app --host 0.0.0.0 --port 4000 --reload &
BOTSCOPE_PID=$!

# Run Spyfind Frontend (foreground — killing this stops everything)
echo "[INFO] Starting Spyfind Frontend (Port 3000)..."
uvicorn frontend.main:app --host 0.0.0.0 --port 3000 --reload

# Cleanup
kill $BACKEND_PID $BOTSCOPE_PID
