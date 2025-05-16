from unittest.mock import patch, Mock
from .test_base import BaseChatCLITest

class TestREPL(BaseChatCLITest):
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