import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from dotenv import load_dotenv

# Walks up from this file's folder and loads the shared agailabs/.env
load_dotenv()

# ==============================================================================
# CONFIGURATION: Simply swap the comment on the PROVIDER variable to switch.
# ==============================================================================

# --- Setup for Amazon Bedrock ---
PROVIDER = "bedrock"
MODEL_ID = "qwen.qwen3-coder-30b-a3b-v1:0" #"amazon.nova-lite-v1:0"       # Your AWS Bedrock model identifier
AWS_REGION = "ap-northeast-1"                 # Your Bedrock AWS Region (e.g., us-east-1)
BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY")

# --- Setup for Local Ollama (Uncomment below to switch) ---
# PROVIDER = "ollama"
# MODEL_ID = "llama3"                    # Your pulled Ollama model ID

# ==============================================================================


def get_llm_response(provider, model, messages):
    """
    Sends raw HTTP requests to Bedrock or Ollama using Python's built-in urllib.
    Zero external packages required.
    """
    try:
        # 1. Determine endpoint URL and security headers based on provider
        if provider == "bedrock":
            # Bedrock Runtime OpenAI-compatible chat endpoint
            url = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {BEDROCK_API_KEY}",
                "Content-Type": "application/json"
            }
        elif provider == "ollama":
            # Local Ollama OpenAI-compatible chat endpoint
            url = "http://localhost:11434/v1/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
        else:
            return f"Error: Unknown provider '{provider}'"

        # 2. Build the unified payload structure
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        # 3. Serialize dict to string and encode to raw UTF-8 bytes for network transfer
        data_bytes = json.dumps(payload).encode("utf-8")
        
        # 4. Create and dispatch the raw HTTP POST request
        req = Request(url, data=data_bytes, headers=headers, method="POST")
        
        with urlopen(req) as response:
            # Read the incoming byte stream and decode back to text string
            response_body = response.read().decode("utf-8")
            response_data = json.loads(response_body)
            
            # Both OpenAI-compatible layers share the identical return structure
            return response_data["choices"][0]["message"]["content"]

    except HTTPError as e:
        return f"HTTP error {e.code}: {e.read().decode('utf-8', errors='ignore')}"
    except URLError as e:
        return f"Network connectivity error: {e.reason}"
    except Exception as e:
        return f"Unexpected parsing error: {e}"


def main():
    print(f"=== Pure Standard Library Connection via [{PROVIDER.upper()}] ===")
    print(f"Active Model: {MODEL_ID}")
    print("Type 'quit' to exit the application.\n")

    # Clean, 100% platform-agnostic conversation context
    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Append the message to the session state
            conversation_history.append({"role": "user", "content": user_input})

            # Process request transparently
            assistant_response = get_llm_response(PROVIDER, MODEL_ID, conversation_history)
            print(f"\nLLM: {assistant_response}\n")

            # Store the assistant context to retain conversational history
            conversation_history.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break


if __name__ == "__main__":
    main()
