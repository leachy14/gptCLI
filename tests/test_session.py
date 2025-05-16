from .test_base import BaseChatCLITest
from chat_cli import Session

class TestSession(BaseChatCLITest):
    def test_session_creation(self):
        """Test that a new session is created with correct initial values"""
        self.assertEqual(self.test_session.name, "test_session")
        self.assertEqual(self.test_session.model, "gpt-4o")
        self.assertEqual(len(self.test_session.messages), 1)  # Should have system prompt
        self.assertFalse(self.test_session.enable_web_search)
        self.assertFalse(self.test_session.enable_reasoning_summary)

    def test_session_save_load(self):
        """Test that a session can be saved and loaded correctly"""
        # Add some test messages
        self.test_session.add_user_message("Hello")
        self.test_session.add_assistant_message("Hi there!")
        
        # Save the session
        self.test_session.save()
        
        # Load the session
        loaded_session = Session.load("test_session")
        
        # Verify the loaded session matches the original
        self.assertEqual(loaded_session.name, self.test_session.name)
        self.assertEqual(loaded_session.model, self.test_session.model)
        self.assertEqual(len(loaded_session.messages), len(self.test_session.messages))
        self.assertEqual(loaded_session.enable_web_search, self.test_session.enable_web_search)
        self.assertEqual(loaded_session.enable_reasoning_summary, self.test_session.enable_reasoning_summary) 