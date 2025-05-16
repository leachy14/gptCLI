import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from chat_cli import Session, ChatCLI, OpenAIClientWrapper

class BaseChatCLITest(unittest.TestCase):
    def setUp(self):
        # Create a temporary test directory for sessions
        self.test_sessions_dir = Path.home() / ".chat_cli_sessions_test"
        self.test_sessions_dir.mkdir(exist_ok=True)
        
        # Patch SESSIONS_DIR to use our test directory
        self.sessions_dir_patcher = patch('chat_cli.core.session.Session.SESSIONS_DIR', self.test_sessions_dir)
        self.sessions_dir_patcher.start()
        
        # Patch print to suppress command feedback
        self.print_patcher = patch('builtins.print')
        self.print_patcher.start()
        
        # Mock the OpenAI client
        self.mock_client = Mock()
        self.mock_wrapper = OpenAIClientWrapper(self.mock_client)
        
        # Create a test session
        self.test_session = Session(
            name="test_session",
            model="gpt-4o",
            messages=[],
            enable_web_search=False,
            enable_reasoning_summary=False
        )
        
        # Create ChatCLI instance
        self.chat_cli = ChatCLI(self.test_session, self.mock_wrapper)

    def tearDown(self):
        # Stop the patchers
        self.sessions_dir_patcher.stop()
        self.print_patcher.stop()
        
        # Clean up test session files
        if self.test_sessions_dir.exists():
            for file in self.test_sessions_dir.glob("*.json"):
                file.unlink()
            self.test_sessions_dir.rmdir() 