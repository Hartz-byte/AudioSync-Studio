# AudioSync Studio - Docker Test Script
# Starts the application stack using Docker Compose
# Press Ctrl+C to stop

Write-Output "🚀 Building and Starting AudioSync Studio in Docker..."
Write-Output "ℹ️  Ensure Docker Desktop is running."

# Build and start in detached mode first to ensure ordering
# docker-compose up --build -d

# Attached mode to see logs
docker-compose up --build
