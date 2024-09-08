# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /llm_agents/Database

# Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client

# Copy the Database directory contents into the container at /app/Database
COPY . /llm_agents/Database

# Install any needed packages specified in requirements.txt
COPY requirements.txt /llm_agents/Database/
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5432 available to the world outside this container
EXPOSE 5432

# Run setup_database.py when the container launches
CMD ["sh", "run_setup.sh"]