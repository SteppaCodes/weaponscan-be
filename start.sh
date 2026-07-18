#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Redis server..."
redis-server --port 6379 --daemonize yes

# Wait for Redis to start
until redis-cli -p 6379 ping | grep -q "PONG"; do
  echo "Waiting for Redis to start..."
  sleep 1
done
echo "Redis is ready!"

echo "Running migrations..."
python manage.py migrate --noinput

echo "Starting Celery Worker..."
celery -A weaponscan worker \
  --queues=inference,default \
  --concurrency=1 \
  --loglevel=info \
  --without-heartbeat \
  --without-mingle &

echo "Starting Celery Beat..."
celery -A weaponscan beat --loglevel=info &

echo "Starting Gunicorn on port ${PORT:-8000}..."
exec gunicorn weaponscan.wsgi:application \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers 1 \
  --timeout 120
