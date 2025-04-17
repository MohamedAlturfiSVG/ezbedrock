# ezbedrock/conversation.py
from typing import Optional
from .bedrock_api import BedrockClientWrapper


class Conversation:
    def __init__(self, client: BedrockClientWrapper, system_prompt: Optional[str] = None, max_token_limit: int = 8000):
        """
        Initialize a conversation with a Bedrock client.

        Args:
            client: An initialized BedrockClientWrapper instance
            system_prompt: Optional system instruction for the model
            max_token_limit: Maximum estimated token count to maintain in history
        """
        self.client = client
        self.history = []  # List of (role, content) tuples
        self.system_prompt = system_prompt
        self.max_token_limit = max_token_limit
        self.full_history = []  # Archive of all messages

    def add_message(self, content: str, role: str = "user") -> None:
        """Add a message to the conversation history."""
        message = {"role": role, "content": content}
        self.history.append(message)
        self.full_history.append(message)
        self._manage_context_window()

    def _estimate_token_count(self, text: str) -> int:
        """Roughly estimate token count (4 chars ≈ 1 token)."""
        return len(text) // 4

    def _manage_context_window(self) -> None:
        """Maintain context while keeping token count under limit."""
        # Roughly estimate current token count
        current_tokens = sum(self._estimate_token_count(msg["content"]) for msg in self.history)

        if current_tokens <= self.max_token_limit:
            return  # We're under the limit, do nothing

        # If over limit, implement a sliding window with smart summarization

        # Always keep system messages (they're usually important context)
        system_messages = [msg for msg in self.history if msg["role"] == "system"]
        non_system_messages = [msg for msg in self.history if msg["role"] != "system"]

        # Always keep the most recent exchanges
        keep_count = min(6, len(non_system_messages))  # Keep last 3 exchanges (6 messages)
        recent_messages = non_system_messages[-keep_count:]

        # Determine how many older messages to summarize
        old_messages = non_system_messages[:-keep_count]

        # If we have enough old messages to summarize
        if len(old_messages) >= 4:  # Only summarize if we have at least 4 messages
            summary_text = self._create_summary(old_messages)

            # Create new history with: system messages + summary + recent messages
            self.history = system_messages + [{"role": "system", "content": summary_text}] + recent_messages
        else:
            # Not enough to summarize, just trim to recent messages
            self.history = system_messages + recent_messages

    def _create_summary(self, messages) -> str:
        """Create a summary of previous conversation messages."""
        # Extract the conversation text in a clear format
        conversation_text = "\n\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages])

        # Create a better summarization prompt
        summary_prompt = (
            "You are tasked with summarizing a conversation history. "
            "Create a concise but informative summary that captures the key points, "
            "decisions, and important context from this conversation. "
            "This summary will replace older messages while maintaining context "
            "for the rest of the conversation.\n\n"
            "CONVERSATION HISTORY TO SUMMARIZE:\n"
            f"{conversation_text}"
        )

        # Use a separate model call for summarization with strong constraints
        try:
            summary = self.client.invoke_model(
                summary_prompt,
                max_tokens=300,  # Limit summary length
                temperature=0.2,  # Low temperature for focused summary
                system_prompt="You are an expert at summarizing conversations.",
            )
            return f"CONVERSATION SUMMARY (previous messages): {summary}"
        except Exception as e:
            # Fallback in case summarization fails
            return "Previous messages were summarized but details are not available."

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
        # Merge default params from client with call-specific kwargs
        params = {**self.client.default_params}
        params.update(kwargs)  # Override defaults with call-specific params

        if params:
            inference_params = self.client._prepare_inference_params(**params)
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

    def get_full_history(self):
        """Return the complete conversation history."""
        return self.full_history

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
        self.full_history = []
