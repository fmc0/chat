#!/usr/bin/env python3
"""
llm_chat_test.py - Provider-agnostic LLM connectivity test + interactive chat.

Runs on Windows WSL (Debian). Checks that your API keys / endpoints work for:
    - AWS Bedrock   (provider = "bedrock")
    - Ollama        (provider = "ollama")
    - OpenAI        (provider = "openai")
    - Anthropic     (provider = "anthropic")   <- Claude direct API

All secrets come from environment variables - nothing is hardwired.

----------------------------------------------------------------------------
SETUP (Debian / WSL)
----------------------------------------------------------------------------
    sudo apt update && sudo apt install -y python3 python3-venv python3-pip
    python3 -m venv ~/llmtest-venv
    source ~/llmtest-venv/bin/activate
    pip install requests boto3 python-dotenv      # boto3 only needed for Bedrock

ENVIRONMENT VARIABLES (put in ~/.bashrc, or in a .env file next to this script)
    LLM_PROVIDER            bedrock | ollama | openai | anthropic   (optional)
    LLM_MODEL               override the model name                 (optional)
    LLM_TEMPERATURE         e.g. 0.7                                (optional)
    LLM_TOP_P               e.g. 0.9                                (optional)
    LLM_TOP_K               e.g. 40                                 (optional)
    LLM_MAX_TOKENS          e.g. 1024                               (optional)

    OPENAI_API_KEY          required for provider=openai
    OPENAI_BASE_URL         optional (default https://api.openai.com/v1)

    ANTHROPIC_API_KEY       required for provider=anthropic

    AWS_ACCESS_KEY_ID       required for provider=bedrock (or use AWS_PROFILE)
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN       optional (temporary credentials / SSO)
    AWS_PROFILE             optional (instead of the keys above)
    AWS_REGION              e.g. us-east-1

    OLLAMA_HOST             default http://localhost:11434
                            (If Ollama runs on Windows, not inside WSL, either
                             enable WSL "mirrored" networking or use the Windows
                             host IP: `ip route show default | awk '{print $3}'`)

    GITHUB_TOKEN            optional - for future GitHub API access

RUN
    python3 llm_chat_test.py                 # uses PROVIDER below / LLM_PROVIDER
    python3 llm_chat_test.py ollama          # provider on the command line

CHAT COMMANDS
    quit / exit   end the program
    /reset        clear the conversation history
    /info         show current provider, model and sampling settings
    /github       test the GITHUB_TOKEN by calling the GitHub API
"""

import json
import os
import subprocess
import sys

import requests

# Optional: load a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# CONFIGURATION - edit these strings, or override with environment variables
# ============================================================================
PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")        # bedrock | ollama | openai | anthropic
#PROVIDER = os.getenv("LLM_PROVIDER", "ollama")        # bedrock | ollama | openai | anthropic

# Model name per provider. Change the quoted strings to the model you want.
MODEL_NAMES = {
    "bedrock":   "qwen.qwen3-coder-30b-a3b-v1:0",                  # Bedrock model / inference-profile ID
    "ollama":    "qwen3-coder:30b",                               # must be pulled: `ollama pull qwen3-coder:30b`
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))   # randomness: 0.0 - 1.0
TOP_P       = float(os.getenv("LLM_TOP_P", "0.9"))         # nucleus sampling (Top-P)
TOP_K       = int(os.getenv("LLM_TOP_K", "40"))            # Top-K / "Top-N" sampling
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

SYSTEM_PROMPT = "You are a helpful assistant."
REQUEST_TIMEOUT = 120  # seconds


# ============================================================================
# GITHUB - headers/helpers ready for future use
# ============================================================================
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "llm-chat-test-script",
}
if GITHUB_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def github_get(path, params=None):
    """GET from the GitHub REST API, e.g. github_get('/user') or
    github_get('/repos/OWNER/REPO/contents/README.md')."""
    url = f"{GITHUB_API_URL}{path}"
    resp = requests.get(url, headers=GITHUB_HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def github_test():
    if not GITHUB_TOKEN:
        print("  GITHUB_TOKEN is not set (unauthenticated requests are rate-limited).")
        info = github_get("/rate_limit")
        print(f"  Unauthenticated rate limit: {info['rate']['remaining']}/{info['rate']['limit']}")
        return
    user = github_get("/user")
    print(f"  GitHub OK - authenticated as: {user.get('login')}")


# ============================================================================
# HELPERS
# ============================================================================
def require_env(*names):
    """Exit with a clear message if any required env var is missing."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        sys.exit(f"ERROR: missing environment variable(s): {', '.join(missing)}")


def is_claude(model):
    return "claude" in model.lower() or "anthropic" in model.lower()


def raise_for_status_verbose(resp):
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")


# ============================================================================
# PROVIDER ADAPTERS - all share the same interface:
#     chat(messages: list[{"role": "user"|"assistant", "content": str}]) -> str
# ============================================================================
class LLMClient:
    def __init__(self, model):
        self.model = model

    def chat(self, messages):
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model):
        super().__init__(model)
        default_host = "http://localhost:11434"
        try:  # WSL: Ollama runs on the Windows host, which is the default gateway
            gw = subprocess.run(["ip", "route", "show", "default"],
                                capture_output=True, text=True).stdout.split()
            default_host = f"http://{gw[2]}:11434"
        except Exception:
            pass
        self.host = os.getenv("OLLAMA_HOST", default_host).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host

    def chat(self, messages):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "num_predict": MAX_TOKENS,
            },
        }
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
        raise_for_status_verbose(resp)
        return resp.json()["message"]["content"]


class OpenAIClient(LLMClient):
    def __init__(self, model):
        super().__init__(model)
        require_env("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }

    def chat(self, messages):
        # Note: OpenAI does not support top_k.
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        }
        resp = requests.post(f"{self.base_url}/chat/completions",
                             headers=self.headers, json=payload, timeout=REQUEST_TIMEOUT)
        raise_for_status_verbose(resp)
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicClient(LLMClient):
    def __init__(self, model):
        super().__init__(model)
        require_env("ANTHROPIC_API_KEY")
        self.headers = {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def chat(self, messages):
        # Newer Claude models reject temperature and top_p together,
        # so we send temperature + top_k and skip top_p.
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        }
        resp = requests.post("https://api.anthropic.com/v1/messages",
                             headers=self.headers, json=payload, timeout=REQUEST_TIMEOUT)
        raise_for_status_verbose(resp)
        data = resp.json()
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


class BedrockClient(LLMClient):
    def __init__(self, model):
        super().__init__(model)
        try:
            import boto3
        except ImportError:
            sys.exit("ERROR: boto3 is required for Bedrock:  pip install boto3")
        # Prefer the Bedrock API key (same auth as chat.py); boto3 reads it
        # from AWS_BEARER_TOKEN_BEDROCK.
        if os.getenv("BEDROCK_API_KEY"):
            os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", os.environ["BEDROCK_API_KEY"])
        elif not os.getenv("AWS_PROFILE"):
            require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION") or "ap-northeast-1"
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def chat(self, messages):
        bedrock_msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
        inference = {"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS}
        extra = {}
        if is_claude(self.model):
            extra["top_k"] = TOP_K        # Claude: temperature + top_k (no top_p)
        else:
            inference["topP"] = TOP_P     # other models: temperature + top_p
        kwargs = dict(
            modelId=self.model,
            messages=bedrock_msgs,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig=inference,
        )
        if extra:
            kwargs["additionalModelRequestFields"] = extra
        resp = self.client.converse(**kwargs)
        return "".join(c.get("text", "") for c in resp["output"]["message"]["content"])


PROVIDERS = {
    "ollama": OllamaClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "claude": AnthropicClient,   # alias
    "bedrock": BedrockClient,
}


def get_client(provider):
    provider = provider.lower()
    if provider not in PROVIDERS:
        sys.exit(f"ERROR: unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    key = "anthropic" if provider == "claude" else provider
    model = os.getenv("LLM_MODEL") or MODEL_NAMES[key]
    return PROVIDERS[provider](model)


# ============================================================================
# MAIN - interactive chat loop
# ============================================================================
def print_info(provider, client):
    print(f"  Provider    : {provider}")
    print(f"  Model       : {client.model}")
    print(f"  Temperature : {TEMPERATURE}")
    print(f"  Top-P       : {TOP_P}")
    print(f"  Top-K       : {TOP_K}")
    print(f"  Max tokens  : {MAX_TOKENS}")
    print(f"  GitHub token: {'set' if GITHUB_TOKEN else 'not set'}")


def main():
    provider = sys.argv[1] if len(sys.argv) > 1 else PROVIDER
    client = get_client(provider)

    print("=" * 60)
    print(" LLM connection test")
    print("=" * 60)
    print_info(provider, client)
    print("-" * 60)

    # Quick connectivity check
    print("Testing connection...")
    try:
        reply = client.chat([{"role": "user", "content": "Reply with exactly: CONNECTION OK"}])
        print(f"  LLM replied: {reply.strip()}")
    except Exception as e:
        print(f"  CONNECTION FAILED: {e}")
        print("  Check your API keys, model name, region/host and network.")
        sys.exit(1)

    print("-" * 60)
    print("Chat started. Type 'quit' to exit, '/reset', '/info', '/github'.")
    print("-" * 60)

    history = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if cmd == "/reset":
            history.clear()
            print("  (conversation cleared)")
            continue
        if cmd == "/info":
            print_info(provider, client)
            continue
        if cmd == "/github":
            try:
                github_test()
            except Exception as e:
                print(f"  GitHub FAILED: {e}")
            continue

        history.append({"role": "user", "content": user_input})
        try:
            answer = client.chat(history)
        except Exception as e:
            history.pop()  # drop the failed turn so history stays valid
            print(f"  ERROR: {e}")
            continue
        history.append({"role": "assistant", "content": answer})
        print(f"\n{client.model}: {answer.strip()}")


if __name__ == "__main__":
    main()