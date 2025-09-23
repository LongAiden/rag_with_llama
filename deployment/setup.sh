#!/bin/bash

# Load environment variables
source deployment/.env

# Install PostgreSQL via Homebrew
brew install postgresql pgvector

# Install Python requirements
pip install -r requirements.txt

# Start
brew services start postgresql

# Create database and user
psql postgres << EOF
-- Create database
CREATE DATABASE rag_db;

-- Create user from environment variables
CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO $POSTGRES_USER;

-- Connect to the database
\c rag_db

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Exit
\q
EOF

