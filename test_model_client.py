from src.prompt_engine import build_prompt
from src.model_client import generate_response


review = (
    "The movie was excellent, emotional, "
    "and beautifully acted."
)

prompt = build_prompt(
    dataset_name="IMDb",
    text=review,
    strategy="zero_shot",
)

print("Sending prompt to Qwen...\n")

response_text, response_time = generate_response(prompt)

print("Model response:", response_text)
print(f"Response time: {response_time:.2f} seconds")