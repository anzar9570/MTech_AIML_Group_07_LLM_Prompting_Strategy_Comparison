
import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen2.5:3b",
    "prompt": (
        "Classify this review as Positive or Negative. "
        "Return only the label.\n"
        "Review: The movie was excellent and beautifully acted."
    ),
    "stream": False
}

try:
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    print("Qwen response:", result["response"].strip())

except requests.exceptions.RequestException as error:
    print("Connection error:", error)
