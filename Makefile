.PHONY: env format clean lint

# Environment name - change as needed
ENV_NAME := beer_pong_env

# Create and configure conda environment
env:
	pip install -r requirements.txt
	
# Format code using black and isort
format:
	black .
	isort .

# Run ruff with auto-fix capabilities
lint:
	ruff check --fix src
	ruff format src

# Clean up Python cache files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete

# Default target
all: format lint 