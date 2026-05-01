#!/bin/sh
set -e
export PYTHONPATH=/app
cd /app/backend && alembic -c alembic.ini upgrade head
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
