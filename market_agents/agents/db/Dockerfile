# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /agents/db

# Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client

# Copy the Database directory contents into the container at /app/Database
COPY . /agents/db

# Install any needed packages specified in requirements.txt
COPY requirements.txt /agents/db/
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5432 available to the world outside this container
EXPOSE 5432

# Run setup_database.py when the container launches
CMD ["sh", "run_setup.sh"]


FROM postgres:14

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-14

RUN git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" > /docker-entrypoint-initdb.d/create_extension.sql

RUN cd pgvector && \
    make && \
    make install

# Set environment variable to create extension on database creation
ENV CREATE_EXTENSION_SQL="CREATE EXTENSION IF NOT EXISTS vector;"