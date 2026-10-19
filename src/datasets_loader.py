
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def load_classification_dataset(dataset_name):
    """
    Load IMDb or AG News from Hugging Face.

    Returns:
        train_records: labelled training examples
        test_records: labelled test examples
        class_labels: mapping from numeric labels to names
    """

    if dataset_name == "IMDb":
        dataset = load_dataset("stanfordnlp/imdb")

        class_labels = {
            0: "Negative",
            1: "Positive",
        }

        text_column = "text"
        label_column = "label"

    elif dataset_name == "AG News":
        dataset = load_dataset("fancyzhx/ag_news")

        class_labels = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }

        text_column = "text"
        label_column = "label"

    else:
        raise ValueError(
            "Unsupported dataset. Choose 'IMDb' or 'AG News'."
        )

    train_records = [
        {
            "text": str(row[text_column]),
            "label": int(row[label_column]),
        }
        for row in dataset["train"]
    ]

    test_records = [
        {
            "text": str(row[text_column]),
            "label": int(row[label_column]),
        }
        for row in dataset["test"]
    ]

    return train_records, test_records, class_labels


def get_evaluation_samples(
    test_records,
    sample_size=20,
    random_seed=42,
):
    """
    Select a reproducible, stratified sample from test data.

    Stratification helps preserve class proportions.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    if sample_size > len(test_records):
        raise ValueError(
            f"Requested {sample_size} samples, but only "
            f"{len(test_records)} test examples are available."
        )

    if sample_size < 2:
        return list(test_records[:sample_size])

    df = pd.DataFrame(test_records)

    # A stratified split requires at least two examples per class
    # in the full dataset and enough selected slots per class.
    class_counts = df["label"].value_counts()

    if (
        class_counts.min() >= 2
        and sample_size >= df["label"].nunique()
    ):
        _, selected_df = train_test_split(
            df,
            test_size=sample_size,
            random_state=random_seed,
            stratify=df["label"],
        )
        selected_df = selected_df.sort_index()
        return selected_df.to_dict(orient="records")

    # Fallback for very small or unusually distributed datasets.
    return (
        df.sample(n=sample_size, random_state=random_seed)
        .sort_index()
        .to_dict(orient="records")
    )
