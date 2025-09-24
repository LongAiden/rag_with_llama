# Load environment variables
source deployment/.env

# Install PostgreSQL via Homebrew
brew install postgresql

# Install pgvector via Homebrew
brew install pgvector

# Install Python requirements
pip install -r deployment/requirements.txt

# Start
brew services start postgresql

# Create database and user
psql postgres << EOF
-- Create database
CREATE DATABASE $POSTGRES_DB;

-- Create user from environment variables
CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;

-- Connect to the database
\c $POSTGRES_DB

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Exit
\q
EOF