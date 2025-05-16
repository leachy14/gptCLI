import unittest
import sys
import os

# Add the parent directory to the Python path so we can import the chat_cli module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
from tests.test_session import TestSession
from tests.test_commands import TestCommands
from tests.test_repl import TestREPL
from tests.test_session_switching import TestSessionSwitching

if __name__ == '__main__':
    # Create a test suite with all test cases
    test_suite = unittest.TestSuite()
    
    # Add test cases from each module
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSession))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCommands))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestREPL))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSessionSwitching))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite) 