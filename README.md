# EZBedrock

A simple wrapper for AWS Bedrock API that makes it easy to work with large language models.

## Installation

### Install from GitHub

You can install the package directly from GitHub:

First, make sure you're authenticated with Github CLI. You can do this by running the following command in your terminal:
```bash
gh-auth login
```

Then, you can install the package using pip:
```bash
pip install git+https://github.com/yourusername/ezbedrock.git
```

## Basic Usage
Note: Make sure you're authenticated with AWS CLI and that you have Permissions to access AWS Bedrock.
```python
from ezbedrock import BedrockClientWrapper

# Create a client
bedrock = BedrockClientWrapper(model_id="anthropic.claude-v2")

# Simple text generation
response = bedrock.invoke_model("Tell me a joke")
print(response)

# Create a conversation
conversation = bedrock.create_conversation()
response = conversation.send("What are the three most popular programming languages?")
```

Refer to the examples.ipynb for more detailed usage and examples.