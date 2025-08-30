import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, MagicMock
from typing import Generator, Dict, Any

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from config import Config
from rag_system import RAGSystem
from vector_store import VectorStore
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from session_manager import SessionManager
from document_processor import DocumentProcessor

@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration"""
    config = Config()
    
    # Use in-memory ChromaDB for testing
    config.CHROMA_PATH = ":memory:"
    config.MAX_RESULTS = 5
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Check if we have a real API key or use a mock
    if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        config.ANTHROPIC_API_KEY = "test_api_key_mock"
    
    return config

@pytest.fixture
def temp_docs_dir():
    """Create a temporary directory with test documents"""
    temp_dir = tempfile.mkdtemp()
    
    # Create test course files
    course1_content = """Course: Introduction to Python
Instructor: Jane Doe
Course Link: https://example.com/python-course

Lesson 1: Python Basics
Lesson 2: Data Structures
Lesson 3: Control Flow

This is lesson 1 content about Python basics. Variables and data types are fundamental concepts.
Python is a high-level programming language that supports multiple programming paradigms.

This is lesson 2 content about data structures. Lists, tuples, and dictionaries are essential.
Understanding how to manipulate data structures is crucial for effective programming.

This is lesson 3 content about control flow. Loops and conditionals control program execution.
Mastering control flow allows you to build complex logic in your programs.
"""
    
    course2_content = """Course: Advanced Web Development
Instructor: John Smith  
Course Link: https://example.com/web-course

Lesson 1: HTML Fundamentals
Lesson 2: CSS Styling
Lesson 3: JavaScript Basics

This is lesson 1 content about HTML fundamentals. HTML provides the structure of web pages.
Learning proper HTML semantics is important for accessibility and SEO.

This is lesson 2 content about CSS styling. CSS controls the visual presentation of web pages.
Responsive design principles ensure your websites work on all devices.

This is lesson 3 content about JavaScript basics. JavaScript adds interactivity to web pages.
Understanding DOM manipulation is key to creating dynamic user interfaces.
"""
    
    # Write test files
    with open(os.path.join(temp_dir, "python_course.txt"), "w") as f:
        f.write(course1_content)
    
    with open(os.path.join(temp_dir, "web_course.txt"), "w") as f:
        f.write(course2_content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = Mock()
    
    # Default mock response
    mock_response = Mock()
    mock_response.content = [Mock(type="text", text="Mock AI response")]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def vector_store(test_config):
    """Create a test vector store"""
    return VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )

@pytest.fixture
def document_processor():
    """Create a document processor"""
    return DocumentProcessor()

@pytest.fixture
def session_manager():
    """Create a session manager"""
    return SessionManager()

@pytest.fixture
def search_tool(vector_store):
    """Create a course search tool"""
    return CourseSearchTool(vector_store)

@pytest.fixture
def tool_manager(search_tool):
    """Create a tool manager with registered tools"""
    manager = ToolManager()
    manager.register_tool(search_tool)
    return manager

@pytest.fixture
def ai_generator(test_config, mock_anthropic_client):
    """Create an AI generator with mocked client"""
    generator = AIGenerator(
        api_key=test_config.ANTHROPIC_API_KEY,
        model=test_config.ANTHROPIC_MODEL,
        max_tool_rounds=test_config.MAX_TOOL_ROUNDS
    )
    
    # Replace with mock client for testing
    generator.client = mock_anthropic_client
    
    return generator

@pytest.fixture
def rag_system(test_config, temp_docs_dir):
    """Create a RAG system with test data"""
    system = RAGSystem(test_config)
    
    # Load test documents
    system.add_course_folder(temp_docs_dir, clear_existing=True)
    
    return system

@pytest.fixture
def sample_query_data():
    """Sample data for testing queries"""
    return {
        "valid_query": {
            "query": "What are Python data structures?",
            "session_id": "test_session_123"
        },
        "no_session_query": {
            "query": "How does CSS work?"
        },
        "empty_query": {
            "query": "",
            "session_id": "test_session"
        },
        "long_query": {
            "query": "A" * 1000,  # Very long query
            "session_id": "test_session"
        }
    }

@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "course1": {
            "title": "Introduction to Python",
            "instructor": "Jane Doe", 
            "course_link": "https://example.com/python-course",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Python Basics", "lesson_link": ""},
                {"lesson_number": 2, "lesson_title": "Data Structures", "lesson_link": ""},
                {"lesson_number": 3, "lesson_title": "Control Flow", "lesson_link": ""}
            ]
        },
        "course2": {
            "title": "Advanced Web Development",
            "instructor": "John Smith",
            "course_link": "https://example.com/web-course", 
            "lessons": [
                {"lesson_number": 1, "lesson_title": "HTML Fundamentals", "lesson_link": ""},
                {"lesson_number": 2, "lesson_title": "CSS Styling", "lesson_link": ""},
                {"lesson_number": 3, "lesson_title": "JavaScript Basics", "lesson_link": ""}
            ]
        }
    }

@pytest.fixture
def mock_successful_anthropic_response():
    """Mock a successful Anthropic API response"""
    response = Mock()
    response.content = [Mock(type="text", text="Here's information about Python data structures: Lists, tuples, and dictionaries are essential for storing and organizing data.")]
    response.stop_reason = "end_turn"
    return response

@pytest.fixture
def mock_tool_use_response():
    """Mock an Anthropic API response that uses tools"""
    # First response: tool use
    tool_response = Mock()
    tool_content = Mock()
    tool_content.type = "tool_use"
    tool_content.name = "search_course_content"
    tool_content.input = {"query": "Python data structures"}
    tool_content.id = "tool_test_123"
    tool_response.content = [tool_content]
    tool_response.stop_reason = "tool_use"
    
    # Second response: final answer
    final_response = Mock()
    final_response.content = [Mock(type="text", text="Based on the search results, Python data structures include lists, tuples, and dictionaries.")]
    final_response.stop_reason = "end_turn"
    
    return [tool_response, final_response]

@pytest.fixture
def mock_api_error_response():
    """Mock an Anthropic API error response"""
    from anthropic import APIError
    return APIError("API request failed")

class MockAnthropicResponse:
    """Mock Anthropic API response class for detailed testing"""
    def __init__(self, content_text=None, stop_reason="end_turn", tool_calls=None):
        self.stop_reason = stop_reason
        if tool_calls:
            self.content = tool_calls
        else:
            self.content = [MockContentBlock(content_text or "Test response")]

class MockContentBlock:
    """Mock content block for Anthropic responses"""
    def __init__(self, text=None, tool_use_id=None, tool_name=None, tool_input=None):
        if tool_name:
            self.type = "tool_use"
            self.name = tool_name
            self.input = tool_input or {}
            self.id = tool_use_id or "test_tool_id"
        else:
            self.type = "text"
            self.text = text or "Test response"

@pytest.fixture
def mock_responses():
    """Collection of different mock response types"""
    return {
        "success": MockAnthropicResponse("Successful response"),
        "tool_use": MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[MockContentBlock(
                tool_name="search_course_content",
                tool_input={"query": "test"},
                tool_use_id="test_id"
            )]
        ),
        "final_after_tool": MockAnthropicResponse("Final response after tool use")
    }

# Utility functions for tests
def create_test_chunks(course_title: str, num_chunks: int = 3):
    """Helper function to create test course chunks"""
    chunks = []
    for i in range(num_chunks):
        chunks.append({
            "content": f"This is chunk {i+1} content for {course_title}",
            "course_title": course_title,
            "lesson_number": (i % 3) + 1,
            "chunk_index": i
        })
    return chunks

def assert_valid_query_response(response_data: Dict[str, Any]):
    """Helper function to validate query response structure"""
    assert "answer" in response_data
    assert "sources" in response_data  
    assert "session_id" in response_data
    assert isinstance(response_data["sources"], list)
    assert len(response_data["answer"]) > 0
    assert len(response_data["session_id"]) > 0

def assert_valid_course_stats(stats_data: Dict[str, Any]):
    """Helper function to validate course stats response structure"""
    assert "total_courses" in stats_data
    assert "course_titles" in stats_data
    assert isinstance(stats_data["total_courses"], int)
    assert isinstance(stats_data["course_titles"], list)
    assert stats_data["total_courses"] >= 0