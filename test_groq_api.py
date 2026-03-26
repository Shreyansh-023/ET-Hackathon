import requests
from src.common.config import load_config

cfg = load_config()

# Test Groq API
url = cfg.groq_base_url + "/chat/completions"
headers = {
    "Authorization": f"Bearer {cfg.groq_api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": cfg.groq_model,
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.1,
    "max_tokens": 100,
}

try:
    print("Testing Groq API...")
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
