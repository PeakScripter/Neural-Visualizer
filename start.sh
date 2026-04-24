#!/usr/bin/env bash
# Start both backend and frontend dev servers

cd "$(dirname "$0")"

echo "Starting FastAPI backend on :8000 ..."
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting React dev server on :5173 ..."
cd ../frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend  →  http://localhost:8000"ron
echo "  Frontend →  http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
