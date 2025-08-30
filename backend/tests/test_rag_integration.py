import pytest
import sys
import os

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from rag_system import RAGSystem
from config import Config
import json

class TestRAGIntegration:
    """Test suite for full RAG system integration"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment with full RAG system"""
        config = Config()
        cls.rag_system = RAGSystem(config)
        
        # Check system state
        cls.has_api_key = bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != "your_anthropic_api_key_here")
        cls.course_analytics = cls.rag_system.get_course_analytics()
        
        print(f"Setup: API Key available: {cls.has_api_key}")
        print(f"Setup: Course analytics: {cls.course_analytics}")
    
    def test_system_initialization(self):
        """Test that RAG system initializes properly"""
        assert self.rag_system.vector_store is not None
        assert self.rag_system.ai_generator is not None
        assert self.rag_system.tool_manager is not None
        assert self.rag_system.document_processor is not None
        assert self.rag_system.session_manager is not None
        
        # Check tools are registered
        tool_names = [tool.get_tool_definition()["name"] for tool in self.rag_system.tool_manager.tools.values()]
        print(f"Registered tools: {tool_names}")
        
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        
        print("✓ RAG system initialization is correct")
    
    def test_course_analytics(self):
        """Test course analytics and data availability"""
        analytics = self.rag_system.get_course_analytics()
        
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert isinstance(analytics["total_courses"], int)
        assert isinstance(analytics["course_titles"], list)
        
        print(f"Course analytics: {analytics}")
        
        if analytics["total_courses"] == 0:
            print("⚠ No courses found - this could be the root issue")
        else:
            print("✓ Courses are available in the system")
    
    def test_tool_definitions_available(self):
        """Test that tool definitions are properly available"""
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        
        assert len(tool_definitions) == 2  # Should have both search and outline tools
        
        for tool_def in tool_definitions:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def
        
        print("✓ Tool definitions are available")
    
    def test_direct_tool_execution(self):
        """Test direct tool execution through tool manager"""
        # Test search tool directly
        search_result = self.rag_system.tool_manager.execute_tool(
            "search_course_content", 
            query="introduction"
        )
        print(f"Direct search tool result: {search_result}")
        
        assert search_result is not None
        assert isinstance(search_result, str)
        
        # Test outline tool directly
        if self.course_analytics["total_courses"] > 0:
            course_name = self.course_analytics["course_titles"][0]
            outline_result = self.rag_system.tool_manager.execute_tool(
                "get_course_outline", 
                course_name=course_name
            )
            print(f"Direct outline tool result: {outline_result}")
            
            assert outline_result is not None
            assert isinstance(outline_result, str)
        
        print("✓ Direct tool execution works")
    
    def test_content_query_integration(self):
        """Test full content query integration"""
        if not self.has_api_key:
            print("⚠ Skipping integration test - no API key available")
            return
        
        # Test content-related query
        response, sources = self.rag_system.query("What is MCP?")
        
        print(f"Content query response: {response}")
        print(f"Content query sources: {sources}")
        
        assert response is not None
        assert isinstance(response, str)
        assert isinstance(sources, list)
        
        # Check if response indicates failure
        if "query failed" in response.lower():
            print("❌ Content query failed - this is the reported issue!")
            print(f"Failure response: {response}")
        elif "error" in response.lower():
            print("❌ Content query returned error")
            print(f"Error response: {response}")
        elif len(response.strip()) == 0:
            print("❌ Content query returned empty response")
        else:
            print("✅ Content query succeeded")
    
    def test_outline_query_integration(self):
        """Test full outline query integration"""
        if not self.has_api_key:
            print("⚠ Skipping integration test - no API key available")
            return
        
        if self.course_analytics["total_courses"] == 0:
            print("⚠ No courses available for outline test")
            return
        
        # Test outline-related query
        course_name = self.course_analytics["course_titles"][0]
        response, sources = self.rag_system.query(f"What lessons are in the {course_name} course?")
        
        print(f"Outline query response: {response}")
        print(f"Outline query sources: {sources}")
        
        assert response is not None
        assert isinstance(response, str)
        
        if "query failed" in response.lower():
            print("❌ Outline query failed")
            print(f"Failure response: {response}")
        else:
            print("✅ Outline query succeeded")
    
    def test_session_management(self):
        """Test session management functionality"""
        if not self.has_api_key:
            print("⚠ Skipping session test - no API key available")
            return
        
        session_id = "test_session_123"
        
        # First query with session
        response1, sources1 = self.rag_system.query("Hello", session_id=session_id)
        print(f"First session query: {response1}")
        
        # Check session history
        history = self.rag_system.session_manager.get_conversation_history(session_id)
        print(f"Session history: {history}")
        
        assert history is not None
        
        print("✓ Session management works")
    
    def test_vector_store_content(self):
        """Test vector store content directly"""
        # Check if vector store has content
        course_count = self.rag_system.vector_store.get_course_count()
        course_titles = self.rag_system.vector_store.get_existing_course_titles()
        
        print(f"Vector store course count: {course_count}")
        print(f"Vector store course titles: {course_titles}")
        
        if course_count > 0:
            # Test direct vector search
            search_results = self.rag_system.vector_store.search("introduction")
            print(f"Direct vector search results:")
            print(f"  - Documents: {len(search_results.documents)}")
            print(f"  - Error: {search_results.error}")
            print(f"  - Is empty: {search_results.is_empty()}")
            
            if not search_results.is_empty():
                print(f"  - First document preview: {search_results.documents[0][:100]}...")
                print("✓ Vector store has searchable content")
            else:
                print("⚠ Vector store search returns empty results")
        else:
            print("❌ Vector store has no courses - this is likely the root issue!")
    
    def test_error_propagation(self):
        """Test how errors are propagated through the system"""
        # Test with invalid tool call
        try:
            result = self.rag_system.tool_manager.execute_tool("nonexistent_tool", query="test")
            print(f"Invalid tool result: {result}")
            assert "not found" in result.lower()
            print("✓ Error propagation works for invalid tools")
        except Exception as e:
            print(f"Exception in error propagation test: {e}")

if __name__ == "__main__":
    # Run tests directly
    test = TestRAGIntegration()
    test.setup_class()
    
    print("=== RAG Integration Diagnostic Tests ===")
    try:
        test.test_system_initialization()
        test.test_course_analytics()
        test.test_tool_definitions_available()
        test.test_vector_store_content()
        test.test_direct_tool_execution()
        test.test_content_query_integration()
        test.test_outline_query_integration()
        test.test_session_management()
        test.test_error_propagation()
        print("\n✅ All RAG integration tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()