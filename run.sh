#!/bin/bash

# Helper script for Docker operations

# Display help message
function show_help {
  echo "Dog Breed Classification Docker Helper"
  echo ""
  echo "Usage: $0 [command]"
  echo ""
  echo "Commands:"
  echo "  build       Build all Docker images"
  echo "  train       Run training service"
  echo "  eval        Run evaluation service"
  echo "  infer       Run inference service"
  echo "  all         Run full pipeline (train, eval, infer)"
  echo "  interactive Start interactive shell"
  echo "  clean       Clean up Docker resources"
  echo "  help        Show this help message"
  echo ""
}

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
  echo "Error: Docker and Docker Compose are required."
  echo "Please install them first."
  exit 1
fi

# Process command
case "$1" in
  build)
    echo "🔨 Building Docker images..."
    docker-compose build
    ;;
    
  train)
    echo "🚀 Starting training service..."
    docker-compose up train
    ;;
    
  eval)
    echo "📊 Starting evaluation service..."
    docker-compose up eval
    ;;
    
  infer)
    echo "🔮 Starting inference service..."
    docker-compose up infer
    ;;
    
  all)
    echo "🚀 Running full pipeline..."
    docker-compose up
    ;;
    
  interactive)
    echo "💻 Starting interactive shell..."
    docker-compose run --rm interactive bash
    ;;
    
  clean)
    echo "🧹 Cleaning up Docker resources..."
    docker-compose down --remove-orphans
    echo "Done! To remove images, run: docker-compose down --rmi all"
    ;;
    
  help|*)
    show_help
    ;;
esac 