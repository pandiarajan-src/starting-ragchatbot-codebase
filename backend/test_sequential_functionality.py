#!/usr/bin/env python3
"""
Test sequential tool calling functionality with real scenarios
"""

import os
import sys

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from config import Config
from rag_system import RAGSystem


def test_sequential_scenarios():
    """Test different sequential tool calling scenarios"""
    print("=== Sequential Tool Calling Tests ===\n")

    config = Config()
    rag_system = RAGSystem(config)

    # Check if API key is available
    if (
        not config.ANTHROPIC_API_KEY
        or config.ANTHROPIC_API_KEY == "your_anthropic_api_key_here"
    ):
        print("⚠ No API key - skipping sequential tests")
        return

    print(f"Max tool rounds configured: {config.MAX_TOOL_ROUNDS}")

    test_scenarios = [
        {
            "name": "Course Outline + Content Search",
            "query": "What is lesson 4 about in the MCP course and find related content about that topic?",
            "description": "Should first get course outline, then search for lesson 4 content",
        },
        {
            "name": "Cross-Course Comparison",
            "query": "Compare the introduction sections of the MCP course and the Computer Use course",
            "description": "Should search both courses for introduction content",
        },
        {
            "name": "Simple Single Tool Query",
            "query": "What courses are available?",
            "description": "Should use get_course_outline or search once",
        },
        {
            "name": "Complex Analysis Query",
            "query": "Find all lessons that discuss client-server architecture across all courses",
            "description": "Should search broadly, then potentially follow up with specific searches",
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Query: {scenario['query']}")
        print(f"   Expected: {scenario['description']}")
        print("-" * 80)

        try:
            # Track the query execution
            response, sources = rag_system.query(scenario["query"])

            print(f"Response: {response}")
            print(f"Sources found: {len(sources)}")

            if sources:
                print("Source details:")
                for j, source in enumerate(sources[:3]):  # Show first 3 sources
                    if hasattr(source, "text"):
                        print(f"  {j+1}. {source.text}")
                    else:
                        print(f"  {j+1}. {source}")

            if "error" in response.lower():
                print("⚠ Query returned error response")
            else:
                print("✓ Query executed successfully")

        except Exception as e:
            print(f"❌ Query failed with exception: {e}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_sequential_scenarios()
