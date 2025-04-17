from pydantic import BaseModel
from ezbedrock import BedrockClientWrapper

bedrock = BedrockClientWrapper(model_id="anthropic.claude-v2")


def invoke_model_example():
    """Example of a basic text response from a model"""
    response = bedrock.invoke_model("Tell me a joke")
    return response


def invoke_model_with_kwargs_example():
    """Example showing how to use common kwargs with invoke_model"""
    response = bedrock.invoke_model(
        prompt="Write a concise paragraph about artificial intelligence",
        max_tokens=250,  # Limits response length
        temperature=0.7,  # Controls randomness (higher = more creative)
        top_p=0.95,  # Controls diversity of word choices
    )
    return response


def invoke_model_structured_example():
    """Example using a Pydantic model to enforce response structure"""

    # Define a Pydantic model for structured responses
    class JokeResponse(BaseModel):
        setup: str
        punchline: str
        category: str

    response = bedrock.invoke_model("Tell me a programming joke", response_model=JokeResponse)
    return response


def invoke_model_generic_json_example():
    """Example getting a generic JSON response without a specific schema"""
    response = bedrock.invoke_model("List three programming languages with their key features", structured_output=True)
    return response


if __name__ == "__main__":
    print("Basic text example:")
    print(invoke_model_example())
    print("\n" + "-" * 50 + "\n")

    print("Example with common kwargs:")
    print(invoke_model_with_kwargs_example())
    print("\n" + "-" * 50 + "\n")

    print("Structured response with Pydantic:")
    structured_response = invoke_model_structured_example()
    print(structured_response.model_dump_json(indent=2))
    print("\n" + "-" * 50 + "\n")

    print("Generic JSON response:")
    print(invoke_model_generic_json_example())
