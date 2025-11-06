.PHONY: help install redis-up redis-down server frontend test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	pip install -r requirements-fastapi.txt

install-frontend:  ## Install frontend dependencies
	cd frontend && npm install

redis-up:  ## Start Redis with docker-compose
	docker-compose up -d redis

redis-down:  ## Stop Redis
	docker-compose down

server:  ## Run FastAPI server
	python -m server.main

frontend:  ## Run Vue.js frontend (dev mode)
	cd frontend && npm run dev

test:  ## Run tests
	pytest tests/ -v

clean:  ## Clean cache and logs
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf logs/*.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.env:  ## Create .env from example
	cp .env.example .env

dev: redis-up server  ## Start development environment
