.PHONY: help test test-unit test-integration test-docker test-docker-unit test-docker-integration clean-test coverage

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================
# Docker Testing (Recommended - Consistent Environment)
# ============================================

test-docker: ## Run all tests inside Docker
	docker-compose --profile test up --build test

test-docker-unit: ## Run only unit tests inside Docker
	docker-compose --profile test run --rm test pytest tests/unit -v

test-docker-integration: ## Run only integration tests inside Docker
	docker-compose --profile test run --rm test pytest tests/integration -v

test-docker-coverage: ## Run tests with coverage report inside Docker
	docker-compose --profile test run --rm test pytest --cov=app --cov-report=html --cov-report=term

test-docker-watch: ## Run tests in watch mode (rebuild on code changes)
	docker-compose --profile test run --rm test pytest tests/ -v --ff --tb=short

test-docker-shell: ## Open a shell in the test container
	docker-compose --profile test run --rm test /bin/bash

# ============================================
# Local Testing (Outside Docker)
# ============================================

test: ## Run all tests locally (requires local dependencies)
	pytest tests/ -v

test-unit: ## Run only unit tests locally
	pytest tests/unit -v

test-integration: ## Run only integration tests locally
	pytest tests/integration -v

test-fast: ## Run tests excluding slow tests
	pytest tests/ -v -m "not slow"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw tests/ -- -v

coverage: ## Generate coverage report locally
	pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in tests/htmlcov/index.html"

# ============================================
# Cleanup
# ============================================

clean-test: ## Clean test artifacts
	rm -rf .pytest_cache
	rm -rf tests/htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ============================================
# Docker Environment Management
# ============================================

docker-up: ## Start all services (app, db, redis)
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View logs from all services
	docker-compose logs -f

docker-rebuild: ## Rebuild and restart all services
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

docker-clean: ## Remove all containers, volumes, and images
	docker-compose down -v --rmi all --remove-orphans

# ============================================
# Development
# ============================================

install-dev: ## Install development dependencies locally
	pip install -r deploy/deployment/requirements.txt
	pip install pytest pytest-asyncio pytest-cov pytest-mock reportlab

setup-db: ## Initialize database (run migrations)
	docker-compose up -d postgres
	sleep 5
	@echo "Database is ready!"

# ============================================
# Quick Commands
# ============================================

quick-test: ## Quick smoke test (fast unit tests only)
	pytest tests/unit/test_chunker_factory.py -v
