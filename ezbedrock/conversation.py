# ezbedrock/conversation.py
from typing import Optional
from .bedrock_api import BedrockClientWrapper


class Conversation:
    """A simple conversation manager for maintaining history with Bedrock models."""

    def __init__(self, client: BedrockClientWrapper, system_prompt: Optional[str] = None):
        """
        Initialize a conversation with a Bedrock client.

        Args:
            client: An initialized BedrockClientWrapper instance
            system_prompt: Optional system instruction for the model
        """
        self.client = client
        self.history = []  # List of (role, content) tuples
        self.system_prompt = system_prompt  # Store system prompt as attribute

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
        """
        # Add user message to history
        self.add_message(message, role="user")

        # Create conversation messages
        messages = []

        # Process each message for the API request
        for msg in self.history:
            # Skip "system" role messages since API doesn't support them
            if msg["role"] == "system":
                continue

            messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

        # Apply system prompt to first message if we have one
        if self.system_prompt and messages:
            first_msg = messages[0]
            if first_msg["role"] == "user":
                first_msg["content"] = [
                    {
                        "text": f"<instructions>\n{self.system_prompt}\n</instructions>\n\n{first_msg['content'][0]['text']}"
                    }
                ]

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
