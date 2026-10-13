
CATEGORIES = ["Billing", "Technical", "Delivery", "Account"]


INSTRUCTIONS = """
You are a customer-support complaint classification system.

Classify each complaint into exactly one category:

Billing: Charges, payments, invoices, or refunds.
Technical: Software bugs, application crashes, or website errors.
Delivery: Shipping, tracking, parcels, or delivery problems.
Account: Login, password, profile, or account access problems.

Return only a JSON object in this format:
{"category": "Billing"}

The category must be exactly one of:
Billing, Technical, Delivery, Account.
"""


def select_examples(train_df):
    """Select two training examples per category."""
    examples = []

    for category in CATEGORIES:
        subset = train_df[train_df["label"] == category]
        examples.extend(
            (row["text"], row["label"])
            for _, row in subset.head(2).iterrows()
        )

    return examples


def build_prompt(text, strategy, train_df):
    """Create a prompt for the selected strategy."""
    prompt = INSTRUCTIONS

    if strategy == "zero_shot":
        # Instructions only: no examples.
        pass

    elif strategy == "good_few_shot":
        prompt += "\nCorrectly labeled examples:\n"

        for example, label in select_examples(train_df):
            prompt += (
                f"\nComplaint: {example}\n"
                f"Category: {label}\n"
            )

    elif strategy == "poor_few_shot":
        prompt += "\nExamples with deliberately mismatched labels:\n"

        for example, actual_label in select_examples(train_df):
            # Shift every label to a different category.
            current_index = CATEGORIES.index(actual_label)
            wrong_label = CATEGORIES[
                (current_index + 1) % len(CATEGORIES)
            ]

            prompt += (
                f"\nComplaint: {example}\n"
                f"Category: {wrong_label}\n"
            )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    prompt += (
        f"\nClassify this new complaint:\n{text}\n"
        'Return JSON only, for example {"category": "Billing"}.'
    )

    return prompt
