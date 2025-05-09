import base64
import boto3
import json
# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
)

MODEL_ID = "us.amazon.nova-premier-v1:0"
# Define your system prompt(s).
system_list = [
    {
        "text": "Bạn là một chuyên gia phân tích"
    }
]
# Define a "user" message including both the image and a text prompt.
message_list = [
    {
        "role": "user",
        "content": [
            {
                "video": {
                    "format": "mp4",
                    "source": {
                        "s3Location": {
                            "uri": "s3://test-content-vinamilk/video/video-test.mp4", 
                            "bucketOwner": "63742331****"
                        }
                    }
                }
            },
            {
                "text": "Provide video titles for this clip."
            }
        ]
    }
]
# Configure the inference parameters.
inf_params = {"maxTokens": 300, "topP": 0.1, "topK": 20, "temperature": 0.3}

native_request = {
    "schemaVersion": "messages-v1",
    "messages": message_list,
    "system": system_list,
    "inferenceConfig": inf_params,
}
# Invoke the model and extract the response body.
response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
model_response = json.loads(response["body"].read())
# Pretty print the response JSON.
print("[Full Response]")
print(json.dumps(model_response, indent=2))
# Print the text content for easy readability.
content_text = model_response["output"]["message"]["content"][0]["text"]
print("\n[Response Content Text]")
print(content_text)