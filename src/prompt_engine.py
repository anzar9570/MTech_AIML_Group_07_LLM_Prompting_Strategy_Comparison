
def get_label_names(dataset_name):
    """Return the valid class labels for a dataset."""

    if dataset_name == "IMDb":
        return {
            0: "Negative",
            1: "Positive",
        }

    if dataset_name == "AG News":
        return {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_prompt(
    dataset_name,
    text,
    strategy="zero_shot",
    examples=None,
):
    """
    Build zero-shot, good few-shot, or poor few-shot prompts.

    examples must contain dictionaries with:
        {"text": "...", "label": integer}
    """

    labels = get_label_names(dataset_name)
    label_names = list(labels.values())

    instructions = f"""
You are a text classification system.

Task: classify the input text into exactly one of these categories:
{", ".join(label_names)}

Rules:
- Read the input carefully.
- Return only one category name from the allowed categories.
- Do not explain your answer.
"""

    prompt = instructions.strip()

    if strategy in ("good_few_shot", "poor_few_shot"):
        if not examples:
            raise ValueError(
                f"Examples are required for {strategy}."
            )

        prompt += "\n\nHere are labelled examples:\n"

        for example in examples:
            label = labels[int(example["label"])]

            # Both few-shot strategies keep the demonstrations' original labels.
            # Poor few-shot differs by selecting lower-similarity examples.
            prompt += (
                f"\nText: {example['text'][:700]}\n"
                f"Category: {label}\n"
            )

    elif strategy != "zero_shot":
        raise ValueError(f"Unknown strategy: {strategy}")

    prompt += (
        "\n\nClassify this new text.\n"
        f"Text: {text[:3000]}\n"
        "Category:"
    )

    return prompt


def build_custom_prompt(
    dataset_name,
    text,
    custom_instructions,
    strategy="zero_shot",
    examples=None,
):
    """Build a user-customized prompt with an optional demonstration strategy.

    strategy can be zero_shot, good_few_shot, or poor_few_shot. The evaluator
    chooses relevant/less-relevant examples; labels are always the original
    labels from the training set.
    """
    labels = get_label_names(dataset_name)
    label_names = list(labels.values())
    instructions = str(custom_instructions or "").strip()

    if not instructions:
        raise ValueError("Please enter custom prompt instructions.")
    if strategy not in ("zero_shot", "good_few_shot", "poor_few_shot"):
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy in ("good_few_shot", "poor_few_shot") and not examples:
        raise ValueError(f"Examples are required for {strategy}.")

    prompt = (
        "You are a text classification system.\n\n"
        f"Allowed categories: {', '.join(label_names)}\n\n"
        "User-provided instructions:\n"
        f"{instructions}\n\n"
        "Important output rule: return exactly one category name from the "
        "allowed categories, with no explanation."
    )

    if strategy in ("good_few_shot", "poor_few_shot"):
        prompt += "\n\nLabelled training examples:\n"
        for example in examples:
            label = labels[int(example["label"])]
            prompt += f"\nText: {str(example['text'])[:700]}\nCategory: {label}\n"

    prompt += (
        "\nClassify this new text.\n"
        f"Text: {str(text)[:3000]}\n\n"
        "Category:"
    )
    return prompt


def parse_prediction(response, dataset_name):
    """
    Extract a valid class label from the model response.
    Returns the numeric label, or None if parsing fails.
    """

    labels = get_label_names(dataset_name)
    response_text = response.strip().lower()

    # Match longer label names first.
    sorted_labels = sorted(
        labels.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for label_id, label_name in sorted_labels:
        if response_text == label_name.lower():
            return label_id

    # Handle a response such as:
    # "Category: Positive" or "The answer is Sports."
    cleaned = response_text.strip(" \t\n\r.,:;\"'`*")

    for label_id, label_name in sorted_labels:
        if cleaned.startswith(label_name.lower()):
            return label_id

    return None
