from src.prompt_engine import build_prompt, parse_prediction


example = {
    "text": "The movie was wonderful and entertaining.",
    "label": 1,
}

review = "The acting was terrible and the story was boring."

strategies = [
    "zero_shot",
    "good_few_shot",
    "poor_few_shot",
]

for strategy in strategies:
    print(f"\n===== {strategy.upper()} =====")

    examples = None

    if strategy == "good_few_shot":
        examples = [example]

    elif strategy == "poor_few_shot":
        # Deliberately incorrect label for testing.
        examples = [
            {
                "text": example["text"],
                "label": 0,
            }
        ]

    prompt = build_prompt(
        dataset_name="IMDb",
        text=review,
        strategy=strategy,
        examples=examples,
    )

    print(prompt)
    print("-" * 50)


print("\n===== PARSER TEST =====")

print("Positive:", parse_prediction("Positive", "IMDb"))
print("Negative:", parse_prediction("Negative", "IMDb"))
print("Invalid:", parse_prediction("Maybe", "IMDb"))