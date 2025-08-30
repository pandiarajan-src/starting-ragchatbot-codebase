#!/usr/bin/env python3
"""
Diagnostic test runner for RAG system
Runs all tests and provides a comprehensive diagnosis of system issues
"""

import os
import sys

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)


def run_test_module(module_name, test_class_name):
    """Run a specific test module and return results"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {module_name}")
    print("=" * 60)

    try:
        # Import and run the test
        module = __import__(module_name)
        test_class = getattr(module, test_class_name)
        test_instance = test_class()
        test_instance.setup_class()

        # Get all test methods
        test_methods = [
            method
            for method in dir(test_instance)
            if method.startswith("test_") and callable(getattr(test_instance, method))
        ]

        passed = 0
        failed = 0

        for method_name in test_methods:
            try:
                print(f"\n--- {method_name} ---")
                getattr(test_instance, method_name)()
                passed += 1
            except Exception as e:
                print(f"❌ {method_name} FAILED: {e}")
                failed += 1

        print(f"\n{module_name} Results: {passed} passed, {failed} failed")
        return passed, failed, None

    except Exception as e:
        print(f"❌ CRITICAL ERROR in {module_name}: {e}")
        import traceback

        traceback.print_exc()
        return 0, 1, str(e)


def main():
    """Run all diagnostic tests and provide summary"""
    print("RAG SYSTEM DIAGNOSTIC TEST SUITE")
    print("=" * 60)
    print("This will run comprehensive tests to identify failing components")

    total_passed = 0
    total_failed = 0
    critical_errors = []

    # Test modules to run
    test_modules = [
        ("test_course_search_tool", "TestCourseSearchTool"),
        ("test_ai_generator", "TestAIGenerator"),
        ("test_rag_integration", "TestRAGIntegration"),
    ]

    # Run each test module
    for module_name, class_name in test_modules:
        passed, failed, error = run_test_module(module_name, class_name)
        total_passed += passed
        total_failed += failed

        if error:
            critical_errors.append(f"{module_name}: {error}")

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")

    if critical_errors:
        print(f"\nCRITICAL ERRORS ({len(critical_errors)}):")
        for error in critical_errors:
            print(f"  ❌ {error}")

    # Diagnosis
    print("\nDIAGNOSIS:")
    print("-" * 20)

    if total_failed == 0:
        print("✅ All tests passed - system appears to be working correctly")
        print("   The 'query failed' issue may be intermittent or frontend-related")
    elif total_failed > total_passed:
        print("❌ Major system issues detected")
        print("   Multiple components are failing - likely root cause issues")
    else:
        print("⚠  Some components failing - investigating specific issues")

    print("\nNext Steps:")
    print("1. Review test output above for specific failure points")
    print("2. Check the most critical failures first")
    print("3. Verify API key configuration and vector store data")
    print("4. Test individual components in isolation")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
