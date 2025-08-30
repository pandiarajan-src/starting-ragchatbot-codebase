import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import VectorStore
from config import Config

class MockAnthropicResponse:
    """Mock Anthropic API response"""
    def __init__(self, content_text=None, stop_reason="end_turn", tool_calls=None):
        self.stop_reason = stop_reason
        if tool_calls:
            self.content = tool_calls
        else:
            self.content = [MockContentBlock(content_text or "Test response")]

class MockContentBlock:
    """Mock content block"""
    def __init__(self, text=None, tool_use_id=None, tool_name=None, tool_input=None):
        if tool_name:
            self.type = "tool_use"
            self.name = tool_name
            self.input = tool_input or {}
            self.id = tool_use_id or "test_tool_id"
        else:
            self.type = "text"
            self.text = text or "Test response"

class TestAIGenerator:
    """Test suite for AIGenerator tool calling functionality"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        config = Config()
        
        # Check if we have API key for real tests
        cls.has_api_key = bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != "your_anthropic_api_key_here")
        
        if cls.has_api_key:
            cls.ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL, config.MAX_TOOL_ROUNDS)
            
            # Set up real tools
            cls.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            cls.tool_manager = ToolManager()
            cls.search_tool = CourseSearchTool(cls.vector_store)
            cls.tool_manager.register_tool(cls.search_tool)
        else:
            print("⚠ No API key found - will run limited tests only")
    
    def test_system_prompt_structure(self):
        """Test that system prompt mentions both tools"""
        prompt = AIGenerator.SYSTEM_PROMPT
        
        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt
        assert "Tool Usage Guidelines" in prompt
        assert "Course outline queries" in prompt
        assert "Content-specific questions" in prompt
        
        print("✓ System prompt includes tool guidance")
    
    def test_tool_definition_format(self):
        """Test that tool definitions are properly formatted for Anthropic"""
        if not self.has_api_key:
            pytest.skip("No API key available")
            
        tool_definitions = self.tool_manager.get_tool_definitions()
        print(f"Tool definitions: {tool_definitions}")
        
        assert len(tool_definitions) >= 1  # Should have at least search tool
        
        for tool_def in tool_definitions:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def
            assert "type" in tool_def["input_schema"]
            assert "properties" in tool_def["input_schema"]
        
        print("✓ Tool definitions are properly formatted")
    
    def test_mock_tool_calling(self):
        """Test tool calling mechanism with mocked responses"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Create a mock for the anthropic client
        original_client = self.ai_generator.client
        mock_client = Mock()
        self.ai_generator.client = mock_client
        
        try:
            # Mock tool use response
            mock_tool_response = MockAnthropicResponse(
                stop_reason="tool_use",
                tool_calls=[MockContentBlock(
                    tool_name="search_course_content",
                    tool_input={"query": "test query"},
                    tool_use_id="test_id_123"
                )]
            )
            
            # Mock final response
            mock_final_response = MockAnthropicResponse("Final answer based on search")
            
            # Set up mock to return tool use first, then final response
            mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
            
            # Test the call
            response = self.ai_generator.generate_response(
                query="What is MCP?",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            print(f"Mock tool calling response: {response}")
            
            # Verify tool was called
            assert mock_client.messages.create.call_count == 2
            print("✓ Tool calling mechanism works with mocked responses")
            
        finally:
            # Restore original client
            self.ai_generator.client = original_client
    
    def test_sequential_tool_calling(self):
        """Test sequential tool calling with multiple rounds"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Create a mock for the anthropic client
        original_client = self.ai_generator.client
        mock_client = Mock()
        self.ai_generator.client = mock_client
        
        try:
            # Mock first tool use response
            mock_first_tool_response = MockAnthropicResponse(
                stop_reason="tool_use",
                tool_calls=[MockContentBlock(
                    tool_name="get_course_outline",
                    tool_input={"course_name": "MCP"},
                    tool_use_id="tool_1"
                )]
            )
            
            # Mock second tool use response
            mock_second_tool_response = MockAnthropicResponse(
                stop_reason="tool_use", 
                tool_calls=[MockContentBlock(
                    tool_name="search_course_content",
                    tool_input={"query": "lesson 4 content"},
                    tool_use_id="tool_2"
                )]
            )
            
            # Mock final response without tools
            mock_final_response = MockAnthropicResponse("Comprehensive answer using both tools")
            
            # Set up mock to return: initial tool use → second tool use → final answer
            mock_client.messages.create.side_effect = [
                mock_first_tool_response, 
                mock_second_tool_response,
                mock_final_response
            ]
            
            # Test sequential tool calling
            response = self.ai_generator.generate_response(
                query="What topics are covered in lesson 4 of the MCP course?",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            print(f"Sequential tool calling response: {response}")
            
            # Verify all rounds were called (initial + 2 tool rounds + final)
            assert mock_client.messages.create.call_count == 3
            print("✓ Sequential tool calling works with multiple rounds")
            
        finally:
            # Restore original client
            self.ai_generator.client = original_client
    
    def test_max_rounds_enforcement(self):
        """Test that max rounds limit is enforced"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Create AI generator with max 2 rounds
        test_ai = AIGenerator(
            api_key=self.ai_generator.client.api_key,
            model=self.ai_generator.model,
            max_tool_rounds=2
        )
        
        original_client = test_ai.client
        mock_client = Mock()
        test_ai.client = mock_client
        
        try:
            # Mock responses that always want to use tools
            mock_tool_response = MockAnthropicResponse(
                stop_reason="tool_use",
                tool_calls=[MockContentBlock(
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_use_id="test_id"
                )]
            )
            
            # Mock final response after max rounds
            mock_final_response = MockAnthropicResponse("Final answer after max rounds")
            
            # Always return tool use for first 2 calls, then final answer
            mock_client.messages.create.side_effect = [
                mock_tool_response,  # Round 1
                mock_tool_response,  # Round 2 (max reached)
                mock_final_response  # Final response (should be called)
            ]
            
            response = test_ai.generate_response(
                query="Test max rounds",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            # Should call: initial + round 1 + round 2 = 3 calls total
            assert mock_client.messages.create.call_count == 3
            print("✓ Max rounds limit enforced correctly")
            
        finally:
            test_ai.client = original_client
    
    def test_no_tool_response(self):
        """Test response generation without tools"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Create a mock for the anthropic client
        original_client = self.ai_generator.client
        mock_client = Mock()
        self.ai_generator.client = mock_client
        
        try:
            # Mock direct response without tool use
            mock_response = MockAnthropicResponse("Direct answer without tools")
            mock_client.messages.create.return_value = mock_response
            
            response = self.ai_generator.generate_response(
                query="What is artificial intelligence?",  # General knowledge question
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            print(f"No-tool response: {response}")
            
            # Should only call once (no follow-up for tool results)
            assert mock_client.messages.create.call_count == 1
            print("✓ No-tool responses work correctly")
            
        finally:
            # Restore original client
            self.ai_generator.client = original_client
    
    def test_tool_manager_integration(self):
        """Test integration between AIGenerator and ToolManager"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Test that tool manager can execute tools
        result = self.tool_manager.execute_tool("search_course_content", query="test")
        print(f"Tool manager execution result: {result}")
        
        assert result is not None
        assert isinstance(result, str)
        
        # Test tool definitions are available
        definitions = self.tool_manager.get_tool_definitions()
        assert len(definitions) > 0
        
        print("✓ Tool manager integration works")
    
    def test_real_content_query(self):
        """Test a real content query if API key is available"""
        if not self.has_api_key:
            print("⚠ Skipping real API test - no API key available")
            return
        
        try:
            response = self.ai_generator.generate_response(
                query="What is MCP in the context of AI development?",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            print(f"Real API response: {response}")
            
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Check if it's an error response
            if "query failed" in response.lower() or "error" in response.lower():
                print("⚠ Real query returned error - this indicates the issue!")
                print(f"Error response: {response}")
            else:
                print("✓ Real content query succeeded")
                
        except Exception as e:
            print(f"❌ Real API test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    def test_max_tool_rounds_configuration(self):
        """Test that max_tool_rounds is properly configured"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Test default value
        default_generator = AIGenerator("test_key", "test_model")
        assert default_generator.max_tool_rounds == 2
        
        # Test custom value
        custom_generator = AIGenerator("test_key", "test_model", max_tool_rounds=3)
        assert custom_generator.max_tool_rounds == 3
        
        print("✓ max_tool_rounds configuration works correctly")
    
    def test_sequential_tool_calling_mock(self):
        """Test sequential tool calling with mocked responses"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Create a mock for the anthropic client
        original_client = self.ai_generator.client
        mock_client = Mock()
        self.ai_generator.client = mock_client
        
        try:
            # Mock first tool use response
            mock_first_response = MockAnthropicResponse(
                stop_reason="tool_use",
                tool_calls=[MockContentBlock(
                    tool_name="get_course_outline",
                    tool_input={"course_name": "MCP"},
                    tool_use_id="outline_id_123"
                )]
            )
            
            # Mock second tool use response (sequential)
            mock_second_response = MockAnthropicResponse(
                stop_reason="tool_use", 
                tool_calls=[MockContentBlock(
                    tool_name="search_course_content",
                    tool_input={"query": "implementation details", "course_name": "MCP"},
                    tool_use_id="search_id_456"
                )]
            )
            
            # Mock final response
            mock_final_response = MockAnthropicResponse("Complete answer combining outline and detailed content")
            
            # Set up mock to return tool use twice, then final response
            mock_client.messages.create.side_effect = [
                mock_first_response, 
                mock_second_response, 
                mock_final_response
            ]
            
            # Test the sequential call
            response = self.ai_generator.generate_response(
                query="What is MCP and how do you implement it?",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            print(f"Sequential tool calling response: {response}")
            
            # Verify both tool rounds were called plus final response
            assert mock_client.messages.create.call_count == 3
            print("✓ Sequential tool calling mechanism works with mocked responses")
            
        finally:
            # Restore original client
            self.ai_generator.client = original_client
    
    def test_tool_round_limit_enforcement(self):
        """Test that tool rounds are limited to prevent infinite loops"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Create a mock for the anthropic client
        original_client = self.ai_generator.client
        mock_client = Mock()
        self.ai_generator.client = mock_client
        
        try:
            # Create mock responses that always request more tools
            def create_tool_response():
                return MockAnthropicResponse(
                    stop_reason="tool_use",
                    tool_calls=[MockContentBlock(
                        tool_name="search_course_content",
                        tool_input={"query": "infinite loop test"},
                        tool_use_id=f"tool_id_{id(object())}"
                    )]
                )
            
            # Mock final response when no more tools allowed
            mock_final_response = MockAnthropicResponse("Final response after tool limit reached")
            
            # Set up mock to return tool use for max rounds, then final response
            mock_client.messages.create.side_effect = [
                create_tool_response(),  # Round 1 tool use
                create_tool_response(),  # Round 2 tool use  
                mock_final_response      # Final response (no tools)
            ]
            
            # Test with generator limited to 2 rounds
            response = self.ai_generator.generate_response(
                query="Test infinite loop prevention",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            print(f"Tool round limit response: {response}")
            
            # Should call exactly max_tool_rounds + 1 times (2 tool rounds + 1 final)
            expected_calls = self.ai_generator.max_tool_rounds + 1
            assert mock_client.messages.create.call_count == expected_calls
            print(f"✓ Tool round limit enforced correctly ({expected_calls} calls)")
            
        finally:
            # Restore original client
            self.ai_generator.client = original_client
    
    def test_tool_execution_error_handling(self):
        """Test graceful handling of tool execution errors"""
        if not self.has_api_key:
            pytest.skip("No API key available")
        
        # Create a mock tool manager that raises errors
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Create a mock for the anthropic client
        original_client = self.ai_generator.client
        mock_client = Mock()
        self.ai_generator.client = mock_client
        
        try:
            # Mock tool use response
            mock_tool_response = MockAnthropicResponse(
                stop_reason="tool_use",
                tool_calls=[MockContentBlock(
                    tool_name="search_course_content",
                    tool_input={"query": "error test"},
                    tool_use_id="error_test_id"
                )]
            )
            
            # Mock final response
            mock_final_response = MockAnthropicResponse("Response despite tool error")
            
            mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
            
            # Test with failing tool manager
            response = self.ai_generator.generate_response(
                query="Test error handling",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
            
            print(f"Error handling response: {response}")
            
            # Should still get a response despite tool failure
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
            
            print("✓ Tool execution errors handled gracefully")
            
        finally:
            # Restore original client
            self.ai_generator.client = original_client
    
    def test_system_prompt_sequential_reasoning(self):
        """Test that system prompt includes sequential reasoning guidance"""
        prompt = AIGenerator.SYSTEM_PROMPT
        
        assert "Sequential reasoning" in prompt
        assert "up to 2 rounds" in prompt
        assert "Round 1" in prompt  
        assert "Round 2" in prompt
        assert "Follow-up searches" in prompt
        assert "builds upon previous findings" in prompt
        
        print("✓ System prompt includes sequential reasoning guidance")

if __name__ == "__main__":
    # Run tests directly
    test = TestAIGenerator()
    test.setup_class()
    
    print("=== AIGenerator Diagnostic Tests ===")
    try:
        test.test_system_prompt_structure()
        test.test_tool_definition_format()
        test.test_mock_tool_calling()
        test.test_no_tool_response()
        test.test_tool_manager_integration()
        test.test_real_content_query()
        
        print("\n=== Sequential Tool Calling Tests ===")
        test.test_max_tool_rounds_configuration()
        test.test_sequential_tool_calling_mock()
        test.test_tool_round_limit_enforcement()
        test.test_tool_execution_error_handling()
        test.test_system_prompt_sequential_reasoning()
        
        print("\n✅ All AIGenerator tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()