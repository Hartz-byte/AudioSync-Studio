@echo off
echo Starting AudioSync Studio...

echo Starting Backend Server...
start "AudioSync Backend" cmd /k "call venv\Scripts\activate && python backend/server.py"

echo Starting Frontend Dev Server...
start "AudioSync Frontend" cmd /k "cd frontend && npm run dev"

echo Done! Access the app at http://localhost:5173
