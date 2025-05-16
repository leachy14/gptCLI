from .test_base import BaseChatCLITest
from chat_cli import Session

class TestSessionSwitching(BaseChatCLITest):
    def test_new_session_command(self):
        """Test creating a new session"""
        # Create a new session
        self.chat_cli.handle_command("/new test_session_2")
        
        # Verify the new session was created
        self.assertEqual(self.chat_cli.session.name, "test_session_2")
        self.assertEqual(len(self.chat_cli.session.messages), 1)  # Should only have system prompt

    def test_switch_session_command(self):
        """Test switching between sessions"""
        # Create and save a second session
        session2 = Session(
            name="test_session_2",
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello from session 2"}]
        )
        session2.save()
        
        # Switch to the second session
        self.chat_cli.handle_command("/switch test_session_2")
        
        # Verify we switched to the correct session
        self.assertEqual(self.chat_cli.session.name, "test_session_2")
        self.assertEqual(self.chat_cli.session.model, "gpt-4.1")
        self.assertEqual(len(self.chat_cli.session.messages), 2)  # System prompt + user message 