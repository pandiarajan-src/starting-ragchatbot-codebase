import pytest
import sys
import os

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from search_tools import CourseSearchTool
from vector_store import VectorStore, SearchResults
from config import Config

class TestCourseSearchTool:
    """Test suite for CourseSearchTool execute method"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment with real VectorStore"""
        config = Config()
        cls.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        cls.search_tool = CourseSearchTool(cls.vector_store)
        
        # Test if we have any data
        cls.course_count = cls.vector_store.get_course_count()
        cls.existing_courses = cls.vector_store.get_existing_course_titles()
        print(f"Setup: Found {cls.course_count} courses: {cls.existing_courses}")
    
    def test_tool_definition(self):
        """Test that tool definition is properly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        assert "name" in definition
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert "properties" in definition["input_schema"]
        assert "query" in definition["input_schema"]["properties"]
        assert "required" in definition["input_schema"]
        assert "query" in definition["input_schema"]["required"]
        
        print("✓ Tool definition is properly formatted")
    
    def test_execute_basic_search(self):
        """Test basic search functionality"""
        if self.course_count == 0:
            pytest.skip("No courses available for testing")
            
        result = self.search_tool.execute(query="introduction")
        print(f"Basic search result: {result}")
        
        # Should not be empty or error
        assert result is not None
        assert isinstance(result, str)
        assert "error" not in result.lower() or "no relevant content found" in result.lower()
        
        print("✓ Basic search executes without critical errors")
    
    def test_execute_with_course_filter(self):
        """Test search with course name filter"""
        if self.course_count == 0:
            pytest.skip("No courses available for testing")
            
        # Use first available course
        course_name = self.existing_courses[0] if self.existing_courses else "MCP"
        result = self.search_tool.execute(query="introduction", course_name=course_name)
        print(f"Course-filtered search result: {result}")
        
        assert result is not None
        assert isinstance(result, str)
        
        print("✓ Course-filtered search executes")
    
    def test_execute_with_lesson_filter(self):
        """Test search with lesson number filter"""
        if self.course_count == 0:
            pytest.skip("No courses available for testing")
            
        result = self.search_tool.execute(query="introduction", lesson_number=1)
        print(f"Lesson-filtered search result: {result}")
        
        assert result is not None
        assert isinstance(result, str)
        
        print("✓ Lesson-filtered search executes")
    
    def test_execute_invalid_course(self):
        """Test search with non-existent course"""
        result = self.search_tool.execute(query="introduction", course_name="NonExistentCourse123")
        print(f"Invalid course search result: {result}")
        
        assert result is not None
        assert isinstance(result, str)
        assert "no course found" in result.lower() or "no relevant content found" in result.lower()
        
        print("✓ Invalid course handled properly")
    
    def test_execute_empty_query(self):
        """Test search with empty query"""
        result = self.search_tool.execute(query="")
        print(f"Empty query result: {result}")
        
        assert result is not None
        assert isinstance(result, str)
        
        print("✓ Empty query handled")
    
    def test_vector_store_connectivity(self):
        """Test that vector store is accessible and has data"""
        print(f"Course count: {self.course_count}")
        print(f"Existing courses: {self.existing_courses}")
        
        if self.course_count > 0:
            # Test direct vector store search
            search_results = self.vector_store.search("introduction")
            print(f"Direct vector store search results: {search_results}")
            print(f"Results empty: {search_results.is_empty()}")
            print(f"Results error: {search_results.error}")
            print(f"Documents count: {len(search_results.documents)}")
            
            assert search_results is not None
            print("✓ Vector store is accessible")
        else:
            print("⚠ No courses found in vector store")
    
    def test_sources_tracking(self):
        """Test that sources are tracked properly"""
        if self.course_count == 0:
            pytest.skip("No courses available for testing")
            
        # Clear previous sources
        self.search_tool.last_sources = []
        
        result = self.search_tool.execute(query="introduction")
        sources = self.search_tool.last_sources
        
        print(f"Tracked sources: {sources}")
        
        assert isinstance(sources, list)
        
        print("✓ Sources tracking works")

if __name__ == "__main__":
    # Run tests directly
    test = TestCourseSearchTool()
    test.setup_class()
    
    print("=== CourseSearchTool Diagnostic Tests ===")
    try:
        test.test_tool_definition()
        test.test_vector_store_connectivity()
        test.test_execute_basic_search()
        test.test_execute_with_course_filter()
        test.test_execute_with_lesson_filter()
        test.test_execute_invalid_course()
        test.test_execute_empty_query()
        test.test_sources_tracking()
        print("\n✅ All CourseSearchTool tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()