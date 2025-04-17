# ezbedrock/conversation.py
from typing import Dict, List, Any

from .bedrock_api import BedrockClientWrapper


class Conversation:
    """A simple conversation manager for maintaining history with Bedrock models."""

    def __init__(self, client: BedrockClientWrapper):
        """
        Initialize a conversation with a Bedrock client.

        Args:
            client: An initialized BedrockClientWrapper instance
        """
        self.client = client
        self.history = []  # List of (role, content) tuples

    def add_message(self, content: str, role: str = "user") -> None:
        """
        Add a message to the conversation history.

        Args:
            content: The message content
            role: The role of the message sender (default: "user")
        """
        self.history.append({"role": role, "content": content})

    def send(self, message: str, **kwargs) -> str:
        """
        Send a message in the context of this conversation.

        Args:
            message: The message to send
            **kwargs: Additional parameters for the model

        Returns:
            The model's response text
        """
        # Add user message to history
        self.add_message(message, role="user")

        # Create conversation messages in the format expected by Bedrock
        messages = []
        for msg in self.history:
            messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

        # Create request body
        request_body = {"messages": messages, "modelId": self.client.model_id}

        # Add inference parameters if specified
        if kwargs:
            inference_params = self.client._prepare_inference_params(**kwargs)
            if inference_params:
                request_body['inferenceConfig'] = inference_params

        # Make API call
        response = self.client.client.converse(**request_body)
        text_response = ""

        if 'output' in response and 'message' in response['output']:
            for content_item in response['output']['message']['content']:
                text_response += content_item['text']

        # Add assistant response to history
        self.add_message(text_response, role="assistant")

        return text_response

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
