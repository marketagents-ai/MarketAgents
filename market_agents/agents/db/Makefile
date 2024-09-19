ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif

setup-pgvector:
    @echo "Setting up pgvector for $(DETECTED_OS)..."
ifeq ($(DETECTED_OS),Darwin)
    @brew install pgvector
    @PG_LIB_DIR=$$(pg_config --pkglibdir) && \
    sudo mkdir -p $$PG_LIB_DIR/../share/extension && \
    sudo cp /opt/homebrew/lib/postgresql@14/vector.so $$PG_LIB_DIR/ && \
    sudo cp /opt/homebrew/share/postgresql@14/extension/vector.control $$PG_LIB_DIR/../share/extension/ && \
    sudo cp /opt/homebrew/share/postgresql@14/extension/vector--*.sql $$PG_LIB_DIR/../share/extension/
    @brew services restart postgresql@14
else ifeq ($(DETECTED_OS),Windows)
    @echo "For Windows, please follow these steps:"
    @echo "1. Download pgvector from https://github.com/pgvector/pgvector/releases"
    @echo "2. Extract the files and copy them to your PostgreSQL installation directory"
    @echo "3. Restart your PostgreSQL service"
    @echo "For detailed instructions, visit: https://github.com/pgvector/pgvector#windows"
else
    @echo "Unsupported operating system. Please install pgvector manually."
endif
    @echo "pgvector setup complete. Please create the extension in your database using: CREATE EXTENSION vector;"