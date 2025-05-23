from unittest.mock import patch
from .test_base import BaseChatCLITest

class TestCommands(BaseChatCLITest):
    def test_model_switching(self):
        """Test that model switching works correctly"""
        # Test switching to a supported model
        self.chat_cli.handle_command("/model gpt-4.1")
        self.assertEqual(self.test_session.model, "gpt-4.1")
        
        # Test switching to an unsupported model
        self.chat_cli.handle_command("/model invalid-model")
        self.assertEqual(self.test_session.model, "gpt-4.1")  # Should not change

    @patch("chat_cli.cli.questionary.select")
    def test_model_interactive(self, mock_select):
        """Interactive model selection uses questionary"""
        mock_select.return_value.ask.return_value = "gpt-4.1"

        self.chat_cli.handle_command("/model")

        mock_select.assert_called_once()
        self.assertEqual(self.test_session.model, "gpt-4.1")

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