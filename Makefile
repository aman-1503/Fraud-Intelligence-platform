# Fraud Intelligence Platform - Makefile
# Convenience commands for development and operations

.PHONY: help install dev test lint build run clean docker-build docker-up docker-down load-test train-model

# Default target
help:
	@echo "Fraud Intelligence Platform - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install      Install all dependencies"
	@echo "  make dev          Run development server"
	@echo "  make test         Run test suite"
	@echo "  make lint         Run linters"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build Docker images"
	@echo "  make docker-up    Start all services"
	@echo "  make docker-down  Stop all services"
	@echo "  make docker-logs  View service logs"
	@echo ""
	@echo "ML:"
	@echo "  make train-model  Train fraud detection model"
	@echo "  make gen-data     Generate synthetic dataset"
	@echo ""
	@echo "Testing:"
	@echo "  make load-test    Run load tests with Locust"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove build artifacts"

# ===========================================
# Development
# ===========================================

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

dev:
	cd backend && uvicorn app.main:app --reload --port 8000

test:
	cd backend && pytest tests/ -v

lint:
	cd backend && black app/ tests/ --check
	cd backend && isort app/ tests/ --check-only
	cd backend && mypy app/

format:
	cd backend && black app/ tests/
	cd backend && isort app/ tests/

# ===========================================
# Docker
# ===========================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --rmi local

# ===========================================
# ML Pipeline
# ===========================================

gen-data:
	cd ml && python generate_dataset.py

train-model:
	cd ml && python train.py

train-sample:
	cd ml && python train.py --sample

# ===========================================
# Load Testing
# ===========================================

load-test:
	cd load-tests && locust -f locustfile.py --host=http://localhost:8000

load-test-headless:
	cd load-tests && locust -f locustfile.py --host=http://localhost:8000 \
		--headless -u 300 -r 50 -t 60s --csv=results

# ===========================================
# Production
# ===========================================

build:
	docker-compose -f docker-compose.yml build

deploy-k8s:
	kubectl apply -f infra/k8s/

# ===========================================
# Cleanup
# ===========================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf backend/.coverage
	rm -rf backend/htmlcov
	rm -rf frontend/dist
	rm -rf frontend/node_modules/.cache

clean-docker:
	docker system prune -f
	docker volume prune -f
