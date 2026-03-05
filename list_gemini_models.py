"""List Gemini models via v1 REST (avoids v1beta)."""
import os
import requests

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY is not set")
    raise SystemExit(1)

url = "https://generativelanguage.googleapis.com/v1/models"
resp = requests.get(url, params={"key": api_key}, timeout=15)
if resp.status_code != 200:
    print(f"Error fetching models ({resp.status_code}): {resp.text[:200]}")
    raise SystemExit(1)

data = resp.json()
models = data.get("models", [])
print(f"Total models: {len(models)}")
for model in models[:20]:
    name = model.get("name")
    methods = model.get("supportedGenerationMethods", [])
    print(f"{name} -> {methods}")
