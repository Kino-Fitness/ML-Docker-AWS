start: ## Start the docker containers
	@echo "Starting the docker containers"
	@docker compose up
	@echo "Containers started - http://localhost:5000"

stop: ## Stop Containers
	@docker compose down

restart: stop start ## Restart Containers

start-bg:  ## Run containers in the background
	@docker compose up -d

build: ## Build Containers
	@docker compose build

ssh: ## SSH into running flaskapi container
	docker compose exec flaskapi bash

bash: ## Get a bash shell into the flaskapi container
	docker compose run --rm --no-deps flaskapi bash

pip-compile: ## Compiles your requirements.in file to requirements.txt
	@docker compose run --rm --no-deps flaskapi pip-compile requirements/requirements.in

requirements: pip-compile build restart  ## Rebuild your requirements and restart your containers