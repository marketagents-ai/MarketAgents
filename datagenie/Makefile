# Set default target
.DEFAULT_GOAL := help

# Variables

## Python
PYTHON := python3
PIP := pip
VENV_NAME := venv

# Help target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  1. install           Install dependencies and set up the environment (should be run first)"
	@echo "  2. run               Run the datagen.py script (should be run second)"
	@echo "  3. clean             Remove the virtual environment and its contents"

# Copy the .env.example file to .env
copy_env:
	cp .env.example .env

# Install dependencies and set up the environment
install: copy_env
	$(PYTHON) -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate && \
	$(PIP) install -r requirements.txt \

# Run the main.py script
run: 
	. $(VENV_NAME)/bin/activate && \
	$(PYTHON) datagen.py --generation_type <generation_type> --num_tasks <num_tasks> --agent_config <agent_config>

# Clean the virtual environment
clean:
	rm -rf $(VENV_NAME)