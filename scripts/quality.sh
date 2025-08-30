#!/bin/bash

# Code quality check script
# Runs all code quality tools in order

set -e  # Exit on any error

echo "🔍 Running code quality checks..."

echo "📋 Checking import sorting with isort..."
uv run isort . --check-only --diff

echo "🎨 Checking code formatting with black..."
uv run black . --check --diff

echo "🧹 Running flake8 linting..."
uv run flake8 .

echo "✅ All code quality checks passed!"