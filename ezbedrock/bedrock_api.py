# ezbedrock/bedrock_api.py
"""
A wrapper library for AWS Bedrock API to simplify model invocation and response handling.
Using the unified Converse API to simplify cross-model interactions.
"""
import json
import re
from typing import Optional, Type, TypeVar, Union, Dict, Any

import boto3
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class BedrockClientWrapper:
    """A wrapper for an AWS Bedrock API client to simplify interactions using the Converse API."""

    def __init__(self, model_id: str = "anthropic.claude-v2", region_name: str = 'us-west-2', **default_params):
        """
        Initialize a Bedrock client wrapper.

        Args:
            model_id: The ID of the model to invoke. Defaults to Claude V2
            region_name: AWS region name
            **default_params: Default parameters to use for all model invocations (temperature, max_tokens, etc.)
        """
        self.client = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.default_params = default_params

    def invoke_model(
        self,
        prompt: str,
        response_model: Optional[Type[T]] = None,
        structured_output: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Union[str, T, Dict[str, Any]]:
        # Merge default params with call-specific kwargs
        params = {**self.default_params}
        params.update(kwargs)  # Override defaults with call-specific params

        # Rest of the method stays the same, but use params instead of kwargs
        enhanced_prompt = self._prepare_prompt(prompt, response_model, structured_output)
        request_body = self._prepare_request_body(enhanced_prompt, system_prompt=system_prompt, **params)
        text_response = self._make_api_call(request_body)

        if response_model or structured_output:
            return self._parse_structured_response(text_response, response_model)

        return text_response

    def create_conversation(self, system_prompt: Optional[str] = None):
        """
        Create a new conversation that maintains history.

        Args:
            system_prompt: Optional system instruction for the model

        Returns:
            A Conversation object linked to this client
        """
        from .conversation import Conversation

        return Conversation(self, system_prompt=system_prompt)

    def _prepare_prompt(self, prompt: str, response_model: Optional[Type[T]], structured_output: bool) -> str:
        """Enhance the prompt with schema instructions if needed."""
        if not (response_model or structured_output):
            return prompt

        if response_model:
            # Get schema from Pydantic model
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)

            # Add schema instructions to prompt
            return (
                f"{prompt}\n\n"
                f"Please respond with a valid JSON object that follows this schema:\n"
                f"```json\n{schema_str}\n```\n"
                f"Ensure the response is a properly formatted JSON object and nothing else."
            )
        else:
            # Generic JSON instruction
            return f"{prompt}\n\n" f"Please respond with a valid JSON object and nothing else."

    def _prepare_request_body(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create the request body with messages and inference parameters."""
        # Create the messages
        messages = []

        # For Claude models, include system instructions in the user message
        if system_prompt:
            # Format the prompt to include system instructions
            formatted_prompt = f"<instructions>\n{system_prompt}\n</instructions>\n\n{prompt}"
            messages.append({"role": "user", "content": [{"text": formatted_prompt}]})
        else:
            # Regular user message
            messages.append({"role": "user", "content": [{"text": prompt}]})

        # Create the request body
        request_body = {"messages": messages, "modelId": self.model_id}

        # Add inference parameters if specified
        if kwargs:
            inference_params = self._prepare_inference_params(**kwargs)
            if inference_params:
                request_body['inferenceConfig'] = inference_params

        return request_body

    def _prepare_inference_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare inference parameters from kwargs."""
        inference_params = {}

        # Map common parameters
        if 'max_tokens' in kwargs:
            inference_params['maxTokens'] = kwargs.pop('max_tokens')

        # Add any remaining parameters directly, converting snake_case to camelCase
        for key, value in kwargs.items():
            camel_key = ''.join([key.split('_')[0]] + [word.capitalize() for word in key.split('_')[1:]])
            inference_params[camel_key] = value

        return inference_params

    def _make_api_call(self, request_body: Dict[str, Any]) -> str:
        """Make the API call and extract text response."""
        response = self.client.converse(**request_body)
        text_response = ""

        if 'output' in response and 'message' in response['output']:
            for content_item in response['output']['message']['content']:
                text_response += content_item['text']

        return text_response

    def _parse_structured_response(
        self, text_response: str, response_model: Optional[Type[T]] = None
    ) -> Union[T, Dict[str, Any]]:
        """Parse and validate JSON response."""
        try:
            # Clean up the response to extract just the JSON part
            json_str = self._extract_json(text_response)
            json_data = json.loads(json_str)

            # Return a validated model if provided
            if response_model:
                return response_model(**json_data)
            return json_data
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse response as valid JSON: {str(e)}\nResponse: {text_response}")

    def _extract_json(self, text: str) -> str:
        """Extract JSON content from a text response that might contain other text."""
        # Look for content between code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1).strip()

        # Try to find JSON-like content (starting with { and ending with })
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json_match.group(1).strip()

        # If no JSON-like content found, return the original text
        return text.strip()
