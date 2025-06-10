#!/bin/bash

# Wait for database to be ready
if [ -n "$DB_HOST" ]; then
  echo "Waiting for database..."
  while ! nc -z $DB_HOST $DB_PORT; do
    sleep 0.5
  done
  echo "Database is ready!"
fi

# Run migrations if needed
# python manage.py db upgrade

# Start the application
exec "$@"