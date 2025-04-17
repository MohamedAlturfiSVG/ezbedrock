# ezbedrock/bedrock_api.py
"""
A wrapper library for AWS Bedrock API to simplify model invocation and response handling.
Using the unified Converse API to simplify cross-model interactions.
"""
import boto3
import json
from typing import Dict, Any, Optional, Union, Iterator


class BedrockClientWrapper:
    """A wrapper for AWS Bedrock API client to simplify interactions using the Converse API."""

    def __init__(self, region_name: str = 'us-west-2', model_id: str = "anthropic.claude-v2"):
        """
        Initialize a Bedrock client wrapper.

        Args:
                region_name: AWS region name
                model_id: The ID of the model to invoke. Defaults to Claude V2
        """
        self.client = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id

    def invoke_model(self, prompt: str, **kwargs) -> str:
        """
        Invoke a Bedrock model with a prompt and return the complete response using Converse API.

        Args:
                prompt: The prompt to send to the model
                **kwargs: Additional parameters for the model

        Returns:
                The model's response text
        """
        # Prepare the messages with user input
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        # Create the request body
        request_body = {"messages": messages, "modelId": self.model_id}

        # Add inference parameters if specified
        if kwargs:
            inference_params = {}

            # Map common parameters
            if 'max_tokens' in kwargs:
                inference_params['maxTokens'] = kwargs.pop('max_tokens')

            # Add any remaining parameters directly
            for key, value in kwargs.items():
                # Convert snake_case to camelCase
                camel_key = ''.join([key.split('_')[0]] + [word.capitalize() for word in key.split('_')[1:]])
                inference_params[camel_key] = value

            if inference_params:
                request_body['inferenceConfig'] = inference_params

        # Make the API call
        response = self.client.converse(**request_body)

        # Parse and return the text response
        if 'output' in response and 'message' in response['output']:
            for content_item in response['output']['message']['content']:
                return content_item['text']

        return ""
