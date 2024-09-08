#!/bin/bash

echo "Database credentials:"
echo "Username: db_user"
echo "Password: db_pwd@123"

# Wait for PostgreSQL to start
until PGPASSWORD=db_pwd@123 psql -h $DB_HOST -U db_user -d postgres -c '\q'; do
  >&2 echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

>&2 echo "PostgreSQL is up - executing command"

# Run the Python script
python setup_database.py

# Keep the container running
tail -f /dev/null