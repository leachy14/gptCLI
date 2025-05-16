import unittest
from unittest.mock import Mock, patch
import json
import os
from pathlib import Path
from chat_cli import Session, ChatCLI, OpenAIClientWrapper, SUPPORTED_MODELS, SESSIONS_DIR

class TestChatCLI(unittest.TestCase):
    def setUp(self):
        # Create a temporary test directory for sessions
        self.test_sessions_dir = Path.home() / ".chat_cli_sessions_test"
        self.test_sessions_dir.mkdir(exist_ok=True)
        
        # Patch SESSIONS_DIR to use our test directory
        self.sessions_dir_patcher = patch('chat_cli.SESSIONS_DIR', self.test_sessions_dir)
        self.sessions_dir_patcher.start()
        
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
        # Stop the patcher
        self.sessions_dir_patcher.stop()
        
        # Clean up test session files
        if self.test_sessions_dir.exists():
            for file in self.test_sessions_dir.glob("*.json"):
                file.unlink()
            self.test_sessions_dir.rmdir()

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

    def test_model_switching(self):
        """Test that model switching works correctly"""
        # Test switching to a supported model
        self.chat_cli.handle_command("/model gpt-4.1")
        self.assertEqual(self.test_session.model, "gpt-4.1")
        
        # Test switching to an unsupported model
        self.chat_cli.handle_command("/model invalid-model")
        self.assertEqual(self.test_session.model, "gpt-4.1")  # Should not change

    def test_clear_command(self):
        """Test that the clear command works correctly"""
        # Add some messages
        self.test_session.add_user_message("Hello")
        self.test_session.add_assistant_message("Hi there!")
        
        # Clear the conversation
        self.chat_cli.handle_command("/clear")
        
        # Should only have the system prompt left
        self.assertEqual(len(self.test_session.messages), 1)
        self.assertEqual(self.test_session.messages[0]["role"], "system")

    @patch('builtins.input')
    def test_repl_basic_interaction(self, mock_input):
        """Test basic REPL interaction with mocked input"""
        # Mock the input to simulate a user message and then exit
        mock_input.side_effect = ["Hello", "/exit"]
        
        # Mock the OpenAI response
        self.mock_client.chat.completions.create.return_value = [
            Mock(choices=[Mock(delta=Mock(content="Hi there!"))])
        ]
        
        # Run the REPL
        self.chat_cli.repl()
        
        # Verify the messages were added
        self.assertEqual(len(self.test_session.messages), 3)  # System prompt + user + assistant
        self.assertEqual(self.test_session.messages[1]["content"], "Hello")
        self.assertEqual(self.test_session.messages[2]["content"], "Hi there!")

    def test_websearch_tool_toggle(self):
        """Test enabling and disabling the web search tool"""
        # Test enabling web search
        self.chat_cli.handle_command("/tool websearch on")
        self.assertTrue(self.test_session.enable_web_search)
        
        # Test disabling web search
        self.chat_cli.handle_command("/tool websearch off")
        self.assertFalse(self.test_session.enable_web_search)
        
        # Test invalid command
        self.chat_cli.handle_command("/tool invalid on")
        self.assertFalse(self.test_session.enable_web_search)  # Should not change

    def test_reasoning_toggle(self):
        """Test enabling and disabling reasoning summaries"""
        # Test enabling reasoning
        self.chat_cli.handle_command("/reasoning on")
        self.assertTrue(self.test_session.enable_reasoning_summary)
        
        # Test disabling reasoning
        self.chat_cli.handle_command("/reasoning off")
        self.assertFalse(self.test_session.enable_reasoning_summary)
        
        # Test invalid command
        self.chat_cli.handle_command("/reasoning invalid")
        self.assertFalse(self.test_session.enable_reasoning_summary)  # Should not change

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

    @patch('builtins.input')
    def test_repl_with_commands(self, mock_input):
        """Test REPL interaction with various commands"""
        # Mock a sequence of inputs: message, model switch, message, exit
        mock_input.side_effect = [
            "Hello",
            "/model gpt-4.1",
            "How are you?",
            "/exit"
        ]
        
        # Mock OpenAI responses
        self.mock_client.chat.completions.create.return_value = [
            Mock(choices=[Mock(delta=Mock(content="Hi there!"))]),
            Mock(choices=[Mock(delta=Mock(content="I'm doing well!"))])
        ]
        
        # Run the REPL
        self.chat_cli.repl()
        
        # Verify the final state
        self.assertEqual(self.chat_cli.session.model, "gpt-4.1")
        self.assertEqual(len(self.chat_cli.session.messages), 5)  # System prompt + 2 user + 2 assistant messages

if __name__ == '__main__':
    unittest.main() 