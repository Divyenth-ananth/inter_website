#!/bin/bash

# --- 1. SETUP ENVIRONMENT ---
echo "--- Step 1: Installing Backend Dependencies ---"
# Check if python3 exists, otherwise try python
PYTHON_CMD=python3
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD=python
fi

$PYTHON_CMD -m pip install -r backend/requirements.txt

echo "--- Step 2: Installing Frontend Dependencies ---"
cd frontend
# Check if pnpm is installed, if not install it
if ! command -v pnpm &> /dev/null; then
    npm install -g pnpm
fi
pnpm install
cd ..

# --- 2. MODEL MANAGEMENT ---
# Requirement VI.C: Model files must be included or downloaded
echo "--- Step 3: Checking/Downloading Model ---"
# Define where the model should live relative to this script
MODEL_DIR="./backend/models/EarthDial_4B_RGB"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model not found. Creating directory..."
    mkdir -p "$MODEL_DIR"
    
    # *** INSERT YOUR DOWNLOAD LOGIC HERE ***
    # Example: wget -O model.zip "YOUR_DROPBOX_OR_DRIVE_LINK"
    # unzip model.zip -d "$MODEL_DIR"
    
    echo "Please ensure model weights are placed in $MODEL_DIR"
else
    echo "Model found at $MODEL_DIR"
fi

# --- 3. EXECUTION ---
echo "--- Step 4: Starting Services ---"

# Start Backend (Background Process)
# We set the MODEL_PATH environment variable here so config.py picks it up
export MODEL_PATH="$MODEL_DIR"
$PYTHON_CMD -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID"

# Start Frontend
cd frontend
pnpm dev &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID"

# Cleanup function to kill processes on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

# Keep script running
wait
