# EZBedrock

A simple wrapper for AWS Bedrock API that makes it easy to work with large language models.

## Key Features

- **Model Flexibility**: Easily switch between any AWS Bedrock model (Claude, Llama, etc.) and customize inference parameters like temperature and token limits
  
- **Simple Text Generation**: Get clean text responses with a single function call - no complex request formatting required
  
- **Structured JSON Output**: Request responses in JSON format for easy parsing and integration with your applications
  
- **Schema Validation**: Define custom Pydantic models to enforce response structure and automatically validate model outputs
  
- **Basic Conversation Management**: Create persistent conversations with automatic context handling that maintains conversation history without exceeding token limits


## Installation

### Install from GitHub

You can install the package directly from GitHub:

First, make sure you're authenticated with Github CLI. You can do this by running the following command in your terminal:
```bash
gh-auth login
```

Then, you can install the package using pip:
```bash
pip install git+https://github.com/MohamedAlturfiSVG/ezbedrock.git
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
```

Refer to the examples.ipynb for more detailed usage and examples.
