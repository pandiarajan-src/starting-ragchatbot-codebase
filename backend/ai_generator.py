from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within courses (lectures, materials, etc.)
2. **get_course_outline**: Get complete course structure with title, instructor, course link, and all lessons with their titles and links

Tool Usage Guidelines:
- **Sequential reasoning**: You can use multiple tool calls across up to 2 rounds to gather comprehensive information
- **Course outline queries** (e.g., "What lessons are in X course?", "Show me the outline of Y"): Use get_course_outline
- **Content-specific questions**: Use search_course_content for detailed educational materials
- **Complex queries**: First use get_course_outline to understand structure, then search_course_content for specific details
- **Follow-up searches**: After initial results, you can search for additional related content if needed
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Sequential Tool Strategy:
- **Round 1**: Get course structure or initial content search
- **Round 2**: Search for specific details, related topics, or complementary information
- Use tool results from previous rounds to inform follow-up searches
- Each round builds upon previous findings for comprehensive answers

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course structure questions**: Use get_course_outline first, optionally follow with content search
- **Course content questions**: Use search_course_content, optionally follow with related searches
- **Complex questions**: Use multiple tool rounds to gather complete information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

When providing course outlines, include:
- Course title and instructor
- Course link if available
- Complete list of lessons with numbers and titles
- Individual lesson links when available

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls with support for sequential tool rounds.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response

        # Track rounds to prevent infinite loops
        for round_num in range(self.max_tool_rounds):
            # Add AI's response to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result,
                            }
                        )
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Tool execution error: {str(e)}",
                            }
                        )

            # Add tool results if any were generated
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Prepare parameters for next API call
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }

            # Include tools for potential follow-up rounds (except last round)
            if round_num < self.max_tool_rounds - 1 and "tools" in base_params:
                next_params["tools"] = base_params["tools"]
                next_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            current_response = self.client.messages.create(**next_params)

            # If no tool use in response, we're done
            if current_response.stop_reason != "tool_use":
                break

        # Return final response text
        return current_response.content[0].text
