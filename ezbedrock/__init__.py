"""
EZBedrock - A simple wrapper for AWS Bedrock API
================================================

This package provides easy access to AWS Bedrock models with simplified interfaces
for both synchronous and streaming invocations.

Example:
    from ezbedrock import BedrockClientWrapper

    bedrock = BedrockClientWrapper(model_id="anthropic.claude-v2")
    response = bedrock.invoke_model("Tell me a joke")
"""

from ezbedrock.bedrock_api import BedrockClientWrapper

__version__ = "0.1.0"
__all__ = ["BedrockClientWrapper"]