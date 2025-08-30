#!/usr/bin/env python3
"""
Fix script to reload course data into vector store
This addresses the empty vector store issue causing query failures
"""

import sys
import os

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from rag_system import RAGSystem
from config import Config

def main():
    """Force reload all course data"""
    print("RAG Vector Store Fix Script")
    print("=" * 40)
    
    try:
        config = Config()
        rag_system = RAGSystem(config)
        
        # Check current state
        print("Current state:")
        analytics = rag_system.get_course_analytics()
        print(f"  Courses: {analytics['total_courses']}")
        print(f"  Titles: {analytics['course_titles']}")
        
        if analytics['total_courses'] == 0:
            print("\n❌ Vector store is empty - this explains the query failures!")
        else:
            print(f"\n✓ Vector store has {analytics['total_courses']} courses")
            # Test search
            search_result = rag_system.vector_store.search("introduction")
            print(f"  Search test: {len(search_result.documents)} documents found")
            if search_result.error:
                print(f"  Search error: {search_result.error}")
        
        # Force reload with clear_existing=True
        print("\n🔄 Force reloading course documents...")
        docs_folder = os.path.join("..", "docs")
        
        if not os.path.exists(docs_folder):
            print(f"❌ Docs folder not found at {docs_folder}")
            return False
        
        # Clear and reload
        total_courses, total_chunks = rag_system.add_course_folder(docs_folder, clear_existing=True)
        
        print(f"✓ Loaded {total_courses} courses with {total_chunks} chunks")
        
        # Verify the fix
        print("\n🔍 Verifying fix...")
        new_analytics = rag_system.get_course_analytics()
        print(f"  New course count: {new_analytics['total_courses']}")
        
        if new_analytics['total_courses'] > 0:
            # Test search functionality
            search_result = rag_system.vector_store.search("introduction")
            print(f"  Search test: {len(search_result.documents)} documents")
            if search_result.error:
                print(f"  Search error: {search_result.error}")
            else:
                print("  ✓ Search working correctly")
            
            # Test tool execution
            tool_result = rag_system.tool_manager.execute_tool("search_course_content", query="introduction")
            if "error" not in tool_result.lower():
                print("  ✓ Search tool working correctly")
            else:
                print(f"  ❌ Search tool still failing: {tool_result[:100]}...")
            
            print("\n✅ Vector store fix completed successfully!")
            print("Content queries should now work properly.")
            return True
        else:
            print("\n❌ Fix failed - vector store still empty")
            return False
            
    except Exception as e:
        print(f"\n❌ Fix script failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nExit code: {0 if success else 1}")
    sys.exit(0 if success else 1)