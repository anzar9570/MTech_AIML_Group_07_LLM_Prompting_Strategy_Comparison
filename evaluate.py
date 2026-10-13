
import json
import re
import time

import ollama
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from prompts import CATEGORIES, build_prompt


# IMPORTANT:
# Change this to the exact model name shown by "ollama list".
MODEL_NAME = "qwen2.5:3b"


def extract_category(response):
    """Extract and validate the predicted category."""
    try:
        data = json.loads(response)
        category = data.get("category", "")

        if category in CATEGORIES:
            return category
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback if the model returns extra text.
    match = re.search(
        r"\b(Billing|Technical|Delivery|Account)\b",
        response,
        flags=re.IGNORECASE
    )

    if match:
        for category in CATEGORIES:
            if match.group(1).lower() == category.lower():
                return category

    # Invalid output counts as an incorrect prediction.
    return "Invalid"


def classify(text, strategy, train_df, retries=3):
    """Classify one complaint using Ollama locally."""
    prompt = build_prompt(text, strategy, train_df)

    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Follow the classification instructions. "
                            "Return a JSON object with a category field."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json",
                options={"temperature": 0}
            )

            content = response["message"]["content"]
            return extract_category(content)

        except Exception:
            if attempt == retries - 1:
                raise

            time.sleep(2 ** attempt)


def evaluate_strategy(train_df, test_df, strategy):
    """Generate predictions and evaluate model performance."""
    y_true = test_df["label"].tolist()
    y_pred = []

    for index, row in test_df.iterrows():
        prediction = classify(
            row["text"],
            strategy,
            train_df
        )

        y_pred.append(prediction)

        print(
            f"{strategy:16} | "
            f"Actual: {row['label']:10} | "
            f"Predicted: {prediction}"
        )

    metrics = {
        "Strategy": strategy,
        "Model": MODEL_NAME,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            labels=CATEGORIES,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            labels=CATEGORIES,
            average="macro",
            zero_division=0
        ),
        "Macro-F1": f1_score(
            y_true,
            y_pred,
            labels=CATEGORIES,
            average="macro",
            zero_division=0
        )
    }

    predictions = test_df.copy()
    predictions["strategy"] = strategy
    predictions["predicted_label"] = y_pred
    predictions["correct"] = (
        predictions["label"] == predictions["predicted_label"]
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=CATEGORIES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CATEGORIES
    )

    return metrics, predictions, report, matrix
