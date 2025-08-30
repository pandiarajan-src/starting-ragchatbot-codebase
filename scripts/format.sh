#!/bin/bash

# Code formatting script
# Automatically formats all Python code

set -e  # Exit on any error

echo "🎨 Formatting Python code..."

echo "📋 Sorting imports with isort..."
uv run isort .

echo "🖤 Formatting code with black..."
uv run black .

echo "✅ Code formatting complete!"