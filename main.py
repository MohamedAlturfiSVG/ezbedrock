from ezbedrock import BedrockClientWrapper

bedrock = BedrockClientWrapper(model_id="anthropic.claude-v2")

def invoke_model_example():
	return bedrock.invoke_model("Tell me a joke")

if __name__ == "__main__":

	print(invoke_model_example())

