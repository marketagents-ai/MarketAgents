#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS system"
    
    # Install pgvector if not already installed
    if ! command_exists brew; then
        echo "Homebrew is not installed. Please install Homebrew first."
        exit 1
    fi
    
    brew install pgvector

    # Set PG_LIB_DIR
    PG_LIB_DIR=$(pg_config --pkglibdir)

    # Create necessary directories
    sudo mkdir -p "$PG_LIB_DIR/../share/extension"

    # Copy files
    sudo cp /opt/homebrew/lib/postgresql@14/vector.so "$PG_LIB_DIR/"
    sudo cp /opt/homebrew/share/postgresql@14/extension/vector.control "$PG_LIB_DIR/../share/extension/"
    sudo cp /opt/homebrew/share/postgresql@14/extension/vector--*.sql "$PG_LIB_DIR/../share/extension/"

    # Restart PostgreSQL
    brew services restart postgresql@14

elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    # Windows
    echo "Detected Windows system"
    
    # Check if PostgreSQL is installed
    if ! command_exists psql; then
        echo "PostgreSQL is not installed or not in PATH. Please install PostgreSQL first."
        exit 1
    }

    # Get PostgreSQL installation directory
    PG_DIR=$(dirname "$(which psql)")
    PG_LIB_DIR="$PG_DIR/lib"
    PG_SHARE_DIR="$PG_DIR/share"

    # Download pgvector files
    curl -LO https://github.com/pgvector/pgvector/releases/latest/download/pgvector.dll
    curl -LO https://github.com/pgvector/pgvector/releases/latest/download/vector.control
    curl -LO https://github.com/pgvector/pgvector/releases/latest/download/vector--0.4.2.sql

    # Copy files
    cp pgvector.dll "$PG_LIB_DIR/"
    cp vector.control "$PG_SHARE_DIR/extension/"
    cp vector--0.4.2.sql "$PG_SHARE_DIR/extension/"

    # Clean up downloaded files
    rm pgvector.dll vector.control vector--0.4.2.sql

    echo "Please restart your PostgreSQL service manually."
else
    echo "Unsupported operating system"
    exit 1
fi

echo "pgvector installation complete. Please create the extension in your database using: CREATE EXTENSION vector;"