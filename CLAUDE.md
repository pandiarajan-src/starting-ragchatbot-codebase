# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## My memory to be stored

# use uv to run python files or add any dependencies

# The vector database has two collections:
    - course_catalog:
        - stores course titles for name resolution
        -  for each course: title, instructor, course_link, lesson_count, lessons_json (list of lessons: lesson_number, lesson_title, lesson_link)
    - course_content:
        - stores text chunks for semantic search
        - metadata for each chunk: course_title, lesson_number, chunk_index

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Install uv package manager (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Code Quality
```bash
# Format code automatically (isort + black)
chmod +x scripts/format.sh && ./scripts/format.sh

# Check code quality without changes (isort + black + flake8)
chmod +x scripts/quality.sh && ./scripts/quality.sh

# Individual commands
uv run isort .              # Sort imports
uv run black .              # Format code
uv run flake8 .             # Lint code
uv run isort . --check-only # Check import sorting
uv run black . --check     # Check formatting
```

### Environment Setup
Required `.env` file in root directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials. The architecture follows a three-tier pattern:

### Core Components

**RAGSystem (`rag_system.py`)** - Central orchestrator that coordinates:
- DocumentProcessor: Chunks course transcripts for vector storage
- VectorStore: ChromaDB integration for semantic search
- AIGenerator: Anthropic Claude API integration with tool calling
- SessionManager: Conversation context persistence
- ToolManager: Orchestrates CourseSearchTool for retrieval

**Data Flow**: User Query → API Endpoint → RAG System → AI Generator (with tools) → Vector Search → Context Assembly → Response Generation

### Key Architectural Patterns

**Tool-Based RAG**: Uses Claude's tool calling to dynamically search vector database rather than pre-retrieving context. The AI decides when and how to search based on the query.

**Session-Aware Conversations**: Each user gets a session_id for maintaining conversation history across multiple queries.

**Chunked Document Processing**: Course transcripts are split into semantic chunks with metadata (course_title, lesson_number, chunk_index) for precise retrieval.

### Backend Structure (`backend/`)

- `app.py` - FastAPI application with 2 endpoints: `/api/query` (POST) and `/api/courses` (GET)
- `models.py` - Pydantic models: Course, Lesson, CourseChunk, QueryRequest, QueryResponse
- `rag_system.py` - Main orchestrator coordinating all components
- `vector_store.py` - ChromaDB wrapper for embedding storage/retrieval
- `ai_generator.py` - Claude API integration with tool calling support
- `document_processor.py` - Text chunking and course metadata extraction
- `session_manager.py` - Conversation history management
- `search_tools.py` - Tool definitions for AI-driven search
- `config.py` - Configuration management

### Frontend Structure (`frontend/`)

Simple web interface that communicates with backend via fetch API:
- `index.html` - Chat interface with course statistics display
- `script.js` - Handles user input, API calls, and response rendering
- `style.css` - UI styling

### Data Models Relationships

```
Course (1) → (many) Lesson
Course (1) → (many) CourseChunk (for vector storage)
QueryRequest → QueryResponse (API contract)
```

### Vector Database Schema

ChromaDB stores CourseChunk objects with:
- `content`: The actual text chunk
- `course_title`: Which course (used for filtering)
- `lesson_number`: Which lesson within course
- `chunk_index`: Position in original document

### Important Implementation Details

**Startup Process**: On server start, automatically processes all `.txt` files in `docs/` folder into vector database.

**Error Handling**: All API endpoints wrapped with try/catch returning HTTP 500 on exceptions.

**Development Features**: Custom DevStaticFiles class adds no-cache headers for frontend development.

**Tool Integration**: AI can access CourseSearchTool which queries vector database and returns relevant chunks with source attribution.

## Application URLs
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`