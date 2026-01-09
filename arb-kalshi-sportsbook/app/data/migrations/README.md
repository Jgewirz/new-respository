# Database Migrations

This directory contains Alembic database migrations.

## Setup

TODO: Initialize Alembic and configure migration environment.

## Usage

```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```
