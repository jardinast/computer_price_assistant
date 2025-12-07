#!/bin/bash

# Computer Price Predictor - Development Runner
# This script starts both the Flask API and React frontend

echo "========================================"
echo "  PriceWise - Computer Price Predictor"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "api/app.py" ] || [ ! -d "frontend" ]; then
    echo "Error: Please run this script from the computer-price-predictor directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Flask API
echo "Starting Flask API on http://localhost:5000..."
cd api
python app.py &
API_PID=$!
cd ..

# Wait a moment for the API to start
sleep 2

# Check if API started successfully
if ! kill -0 $API_PID 2>/dev/null; then
    echo "Error: Failed to start Flask API"
    exit 1
fi

# Start React frontend
echo "Starting React frontend on http://localhost:5173..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "  Servers are running!"
echo "========================================"
echo ""
echo "  API:      http://localhost:5000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop all servers"
echo "========================================"

# Wait for both processes
wait


