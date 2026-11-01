import os
import sys
from dotenv import load_dotenv

# Search upwards from tstllm to locate and read the root .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

print("--- System Baseline Diagnostics ---")
print(f"Python Executable Path: {sys.executable}")

try:
    import langchain
    print(f"✅ LangChain is installed (Version: {langchain.__version__})")
except ImportError:
    print("❌ LangChain is missing from the environment path.")

try:
    import git
    print("✅ GitPython is installed and accessible.")
except ImportError:
    print("❌ GitPython is missing from the environment path.")

print("\n--- LLM Credential Check ---")
print(f"OpenAI Key Configured: {bool(os.environ.get('OPENAI_API_KEY'))}")
print(f"Claude Key Configured: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
print(f"AWS Key Configured:    {bool(os.environ.get('AWS_ACCESS_KEY_ID'))}")
