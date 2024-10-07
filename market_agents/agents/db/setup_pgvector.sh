#!/bin/bash

# Install pgvector if not already installed
brew install pgvector

# Set PG_LIB_DIR
PG_LIB_DIR=$(pg_config --pkglibdir)

# Create necessary directories
sudo mkdir -p $PG_LIB_DIR/../share/extension

# Copy files
sudo cp /opt/homebrew/lib/postgresql@14/vector.so $PG_LIB_DIR/
sudo cp /opt/homebrew/share/postgresql@14/extension/vector.control $PG_LIB_DIR/../share/extension/
sudo cp /opt/homebrew/share/postgresql@14/extension/vector--*.sql $PG_LIB_DIR/../share/extension/

# Restart PostgreSQL
brew services restart postgresql@14

echo "pgvector installation complete. Please create the extension in your database using: CREATE EXTENSION vector;"