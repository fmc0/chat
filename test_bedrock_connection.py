import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

REGION = "ap-northeast-1"
MODEL_ID = "qwen.qwen3-coder-30b-a3b-v1:0"
# Read the Bedrock API key from the environment - never hardcode it.
API_KEY = os.environ.get("BEDROCK_API_KEY")


def main():
    api_key = API_KEY
    if not api_key:
        print("FAIL: set the BEDROCK_API_KEY environment variable first.", file=sys.stderr)
        return 2

    url = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Reply with exactly: Bedrock connection works."}
        ],
        "max_tokens": 32,
        "temperature": 0,
        "stream": False,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print(f"Testing Bedrock model {MODEL_ID} in {REGION}...")
    started = time.monotonic()

    try:
        with urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        message = response_data["choices"][0]["message"]["content"]
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"FAIL: HTTP {error.code}: {details}", file=sys.stderr)
        return 1
    except URLError as error:
        print(f"FAIL: network error: {error.reason}", file=sys.stderr)
        return 1
    except (KeyError, IndexError, json.JSONDecodeError) as error:
        print(f"FAIL: unexpected Bedrock response: {error}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print(f"PASS: {message.strip()}")
    print(f"Response time: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())