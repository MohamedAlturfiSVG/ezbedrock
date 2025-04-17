# ezbedrock/bedrock_api.py
"""
A wrapper library for AWS Bedrock API to simplify model invocation and response handling.
"""
import boto3
import json
from typing import Dict, Any, Optional, Union, Iterator


class BedrockClientWrapper:
	"""A wrapper for AWS Bedrock API client to simplify interactions."""

	def __init__(self, region_name: str = 'us-west-2', model_id: str = "anthropic.claude-v2"):
		"""
        Initialize a Bedrock client wrapper.

        Args:
            region_name: AWS region name
            model_id: The ID of the model to invoke. Defaults to Claude 3.5 V2
        """
		self.client = boto3.client('bedrock-runtime', region_name=region_name)
		self.model_id = model_id

	def invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
		"""
        Invoke a Bedrock model with a prompt and return the complete response.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the model

        Returns:
            The model's response as a dictionary
        """
		body = self._prepare_request_body(self.model_id, prompt, **kwargs)

		response = self.client.invoke_model(
				modelId=self.model_id,
				body=json.dumps(body)
		)

		return self._parse_response(response)

	def invoke_model_with_streaming(self, prompt: str, **kwargs) -> Iterator[Dict[str, Any]]:
		"""
        Invoke a Bedrock model with streaming response.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the model

        Returns:
            An iterator of response chunks
        """
		body = self._prepare_request_body(self.model_id, prompt, **kwargs)

		response = self.client.invoke_model_with_response_stream(
				modelId=self.model_id,
				body=json.dumps(body)
		)

		stream = response.get('body')
		if not stream:
			return

		for event in stream:
			chunk = event.get('chunk')
			if chunk:
				yield json.loads(chunk.get('bytes').decode())

	def _prepare_request_body(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
		"""Prepare the request body based on the model type."""
		# Default parameters
		default_params = {
			'max_tokens_to_sample': kwargs.get('max_tokens', 4000),
		}

		# Model-specific formatting
		if model_id.startswith('anthropic.claude'):
			# Claude models use a specific prompt format
			formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
			body = {
				'prompt': formatted_prompt,
				**default_params
			}
		else:
			# Generic case for other models
			body = {
				'prompt': prompt,
				**default_params
			}

		# Add any other kwargs
		for key, value in kwargs.items():
			if key != 'max_tokens':  # We've already handled this
				body[key] = value

		return body

	def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
		"""Parse the response from the model."""
		response_body = json.loads(response.get('body').read())
		response = response_body.get('completion', [{}])
		return response