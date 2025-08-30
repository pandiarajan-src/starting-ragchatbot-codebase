#!/usr/bin/env python3
"""
Simple diagnostic tests without pytest dependency
Runs basic tests to identify RAG system issues
"""

import os
import sys
import traceback

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)


def test_imports():
    """Test that all modules can be imported"""
    print("=== Testing Imports ===")

    modules_to_test = [
        "config",
        "vector_store",
        "search_tools",
        "ai_generator",
        "rag_system",
    ]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name} - imported successfully")
        except Exception as e:
            print(f"❌ {module_name} - import failed: {e}")
            return False

    print("✅ All imports successful\n")
    return True


def test_vector_store():
    """Test vector store functionality"""
    print("=== Testing Vector Store ===")

    try:
        from config import Config
        from vector_store import VectorStore

        config = Config()
        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        # Test basic functionality
        course_count = vector_store.get_course_count()
        course_titles = vector_store.get_existing_course_titles()

        print(f"Course count: {course_count}")
        print(f"Course titles: {course_titles}")

        if course_count == 0:
            print("❌ CRITICAL: No courses found in vector store!")
            print("   This is likely the root cause of query failures")
            return False

        # Test search functionality
        search_results = vector_store.search("introduction")
        print(f"Search results: {len(search_results.documents)} documents found")
        print(f"Search error: {search_results.error}")
        print(f"Search is empty: {search_results.is_empty()}")

        if search_results.is_empty() and not search_results.error:
            print("⚠ Vector store has courses but search returns empty results")
        elif search_results.error:
            print(f"❌ Vector store search error: {search_results.error}")
        else:
            print("✓ Vector store search working")

        return True

    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        traceback.print_exc()
        return False


def test_search_tool():
    """Test CourseSearchTool directly"""
    print("\n=== Testing CourseSearchTool ===")

    try:
        from config import Config
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        config = Config()
        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        search_tool = CourseSearchTool(vector_store)

        # Test tool definition
        definition = search_tool.get_tool_definition()
        print(f"Tool name: {definition.get('name')}")

        # Test execution
        result = search_tool.execute(query="introduction")
        print(f"Search tool result: {result}")

        if (
            "error" in result.lower()
            and "no relevant content found" not in result.lower()
        ):
            print("❌ Search tool returned error")
            return False
        elif "no relevant content found" in result.lower():
            print("⚠ Search tool found no content (but executed successfully)")
        else:
            print("✓ Search tool executed successfully")

        return True

    except Exception as e:
        print(f"❌ Search tool test failed: {e}")
        traceback.print_exc()
        return False


def test_ai_generator():
    """Test AIGenerator configuration"""
    print("\n=== Testing AIGenerator ===")

    try:
        from ai_generator import AIGenerator
        from config import Config

        config = Config()

        # Check API key
        has_api_key = bool(
            config.ANTHROPIC_API_KEY
            and config.ANTHROPIC_API_KEY != "your_anthropic_api_key_here"
        )
        print(f"Has valid API key: {has_api_key}")

        if not has_api_key:
            print("❌ CRITICAL: No valid Anthropic API key found!")
            print("   Set ANTHROPIC_API_KEY in .env file")
            return False

        ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        print(f"Model: {ai_generator.model}")
        print("✓ AIGenerator initialized successfully")

        return True

    except Exception as e:
        print(f"❌ AI generator test failed: {e}")
        traceback.print_exc()
        return False


def test_rag_system():
    """Test full RAG system"""
    print("\n=== Testing RAG System ===")

    try:
        from config import Config
        from rag_system import RAGSystem

        config = Config()
        rag_system = RAGSystem(config)

        # Test initialization
        print("✓ RAG system initialized")

        # Test analytics
        analytics = rag_system.get_course_analytics()
        print(f"Analytics: {analytics}")

        # Test tools
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        print(f"Available tools: {tool_names}")

        # Test direct tool execution
        search_result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="test"
        )
        print(f"Direct tool execution result: {search_result[:100]}...")

        return True

    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        traceback.print_exc()
        return False


def test_real_query():
    """Test a real query if possible"""
    print("\n=== Testing Real Query ===")

    try:
        from config import Config
        from rag_system import RAGSystem

        config = Config()

        # Check API key first
        if (
            not config.ANTHROPIC_API_KEY
            or config.ANTHROPIC_API_KEY == "your_anthropic_api_key_here"
        ):
            print("⚠ Skipping real query test - no API key")
            return True

        rag_system = RAGSystem(config)

        # Test content query
        print("Testing content query...")
        response, sources = rag_system.query("What is MCP?")

        print(f"Query response: {response}")
        print(f"Query sources: {sources}")

        if "query failed" in response.lower():
            print("❌ CONFIRMED: Query failed - this is the reported issue!")
            return False
        elif not response or len(response.strip()) == 0:
            print("❌ Query returned empty response")
            return False
        else:
            print("✓ Query executed successfully")
            return True

    except Exception as e:
        print(f"❌ Real query test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    print("SIMPLE RAG SYSTEM DIAGNOSTICS")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Vector Store Test", test_vector_store),
        ("Search Tool Test", test_search_tool),
        ("AI Generator Test", test_ai_generator),
        ("RAG System Test", test_rag_system),
        ("Real Query Test", test_real_query),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("✅ All tests passed - system should be working")
    else:
        print("❌ Issues detected - see detailed output above")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
