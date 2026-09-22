.DEFAULT_GOAL := help
COMPOSE ?= docker compose

.PHONY: help init up up-backend down restart logs ps migrate seed create-admin test lint format psql shell revision reset

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

init:  ## Create .env with generated secrets (first-time setup)
	./scripts/init-env.sh

up:  ## Build and start the whole stack (migrations run automatically)
	$(COMPOSE) up -d --build

up-backend:  ## Start everything except the frontend
	$(COMPOSE) up -d --build postgres redis migrate backend worker

down:  ## Stop the stack (keeps data)
	$(COMPOSE) down

restart:  ## Restart application containers
	$(COMPOSE) restart backend worker frontend

logs:  ## Follow logs (make logs s=backend)
	$(COMPOSE) logs -f --tail=100 $(s)

ps:  ## Show service status
	$(COMPOSE) ps

migrate:  ## Apply database migrations
	$(COMPOSE) run --rm migrate

seed:  ## Seed default categories and AI settings (idempotent)
	$(COMPOSE) run --rm backend python -m app.cli.seed

create-admin:  ## Create the first admin (make create-admin email=you@example.com)
	$(COMPOSE) run --rm backend python -m app.cli.create_admin --email "$(email)"

test:  ## Run the backend test-suite against real Postgres + Redis
	$(COMPOSE) run --rm backend pytest

lint:  ## Lint the backend
	$(COMPOSE) run --rm --no-deps backend ruff check .

format:  ## Auto-format the backend
	$(COMPOSE) run --rm --no-deps backend ruff format .

revision:  ## Autogenerate a migration (make revision m="add foo")
	$(COMPOSE) run --rm backend alembic revision --autogenerate -m "$(m)"

psql:  ## Open a psql shell in the database
	$(COMPOSE) exec postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

shell:  ## Shell inside the backend container
	$(COMPOSE) run --rm backend bash

reset:  ## DANGER: stop the stack and DELETE all data volumes
	$(COMPOSE) down -v
