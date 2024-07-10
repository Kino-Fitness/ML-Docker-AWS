# Makefile for Docker Compose operations

# Variables
COMPOSE_FILE = docker-compose.yml
AWS_ACCOUNT_ID = your-aws-account-id
AWS_REGION = your-aws-region
ECR_REPO = your-ecr-repo-name
IMAGE_NAME = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO):latest

# Build and run the Docker container
start:
	docker-compose build
	docker-compose up
	
# Build the Docker image
build:
	docker-compose build

# Run the Docker container
up:
	docker-compose up -d

# Stop the Docker container
down:
	docker-compose down

# View Docker container logs
logs:
	docker-compose logs -f

# SSH into the running container
ssh:
	docker-compose exec model-app /bin/bash

# AWS ECR login
ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

# Tag the Docker image for ECR
tag:
	docker tag model-app:latest $(IMAGE_NAME)

# Push the Docker image to ECR
push:
	docker push $(IMAGE_NAME)

# Deploy to ECR (login, tag, and push)
deploy: ecr-login tag push

# Clean up: Stop and remove the container, and remove the image
clean:
	docker-compose down -v --rmi all

# Run tests (you'll need to define your test command)
test:
	docker-compose run model-app python -m unittest discover tests

# All-in-one command for local development
dev: build up

# All-in-one command for deployment
prod: build test deploy

.PHONY: build up down logs ssh ecr-login tag push deploy clean test dev prod