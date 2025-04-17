from ezbedrock import BedrockClientWrapper

bedrock = BedrockClientWrapper(model_id="anthropic.claude-v2")

def invoke_model_example():
	return bedrock.invoke_model("Tell me a joke")

def invoke_model_streaming_example():
	stream = bedrock.invoke_model_with_streaming("Tell me a joke")

	response_text = ""
	for chunk in stream:
		if "completion" in chunk:
			chunk_text = chunk["completion"]
			response_text += chunk_text
			print(chunk_text, end="", flush=True)

	print("\n")
	return response_text

if __name__ == "__main__":

	#print(invoke_model_example())
	invoke_model_streaming_example()

