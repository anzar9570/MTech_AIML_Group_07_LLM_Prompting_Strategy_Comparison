import random
from pathlib import Path
from datetime import datetime

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.datasets_loader import (
    load_classification_dataset,
    get_evaluation_samples,
)
from src.prompt_engine import build_prompt, build_custom_prompt, parse_prediction
from src.model_client import generate_response


# Directory for saving results
RESULTS_DIR = Path("results")

# Prompting strategies
STRATEGIES = [
    "zero_shot",
    "good_few_shot",
    "poor_few_shot",
]


def prepare_example_candidates(
    train_records,
    class_labels,
    examples_per_class=1,
    random_seed=42,
    candidates_per_class=100,
):
    """
    Prepare a reproducible pool of correctly labelled training examples.

    Only training records are used for demonstrations.
    """
    rng = random.Random(random_seed)
    candidates = []

    for label_id in class_labels:
        class_records = [
            record
            for record in train_records
            if int(record["label"]) == label_id
        ]

        pool_size = min(candidates_per_class, len(class_records))

        if pool_size < examples_per_class:
            raise ValueError(
                f"Not enough training examples for "
                f"{class_labels[label_id]}."
            )

        candidates.extend(
            rng.sample(class_records, pool_size)
        )

    candidate_texts = [
        str(record["text"]) for record in candidates
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
    )

    candidate_matrix = vectorizer.fit_transform(candidate_texts)

    return candidates, vectorizer, candidate_matrix


def select_examples_for_text(
    text,
    candidates,
    vectorizer,
    candidate_matrix,
    class_labels,
    examples_per_class=1,
    most_relevant=True,
):
    """
    Select demonstrations separately for each test text.

    most_relevant=True:
        Select examples with the highest TF-IDF cosine similarity.

    most_relevant=False:
        Select examples with the lowest TF-IDF cosine similarity.

    The original training labels are always preserved.
    """
    query_vector = vectorizer.transform([str(text)])

    similarities = cosine_similarity(
        query_vector,
        candidate_matrix,
    ).flatten()

    selected_examples = []

    for label_id in class_labels:
        class_indices = [
            index
            for index, record in enumerate(candidates)
            if int(record["label"]) == label_id
        ]

        ranked_indices = sorted(
            class_indices,
            key=lambda index: similarities[index],
            reverse=most_relevant,
        )

        for index in ranked_indices[:examples_per_class]:
            selected_examples.append(candidates[index])

    return selected_examples


def calculate_metrics(y_true, y_pred, class_labels):
    """
    Calculate accuracy, macro-precision, macro-recall and macro-F1.

    Invalid predictions are represented by -1 and count as incorrect.
    """
    class_ids = list(class_labels.keys())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0,
        ),
    }


def run_evaluation(
    dataset_names=None,
    sample_size=20,
    random_seed=42,
    examples_per_class=1,
):
    """
    Run zero-shot, good few-shot and poor few-shot evaluations.

    Returns:
        summary_df: Metrics for each dataset and strategy.
        details_df: Individual predictions and their details.
    """
    if dataset_names is None:
        dataset_names = ["IMDb", "AG News"]

    all_summary = []
    all_details = []

    for dataset_name in dataset_names:
        print("\n" + "=" * 60)
        print(f"Dataset: {dataset_name}")
        print("=" * 60)

        # Load dataset
        train_records, test_records, class_labels = (
            load_classification_dataset(dataset_name)
        )

        # Select a fixed evaluation sample
        evaluation_samples = get_evaluation_samples(
            test_records,
            sample_size=sample_size,
            random_seed=random_seed,
        )

        # Prepare labelled training candidates
        candidates, vectorizer, candidate_matrix = (
            prepare_example_candidates(
                train_records=train_records,
                class_labels=class_labels,
                examples_per_class=examples_per_class,
                random_seed=random_seed,
                candidates_per_class=100,
            )
        )

        print("Training examples:", len(train_records))
        print("Testing examples:", len(test_records))
        print("Evaluation sample:", len(evaluation_samples))
        print("Candidate demonstrations:", len(candidates))
        print("Examples per class:", examples_per_class)

        # Run all prompting strategies on the same test samples
        for strategy in STRATEGIES:
            print(f"\nRunning strategy: {strategy}")
            actual_examples_per_class = 0 if strategy == "zero_shot" else examples_per_class

            y_true = []
            y_pred = []

            for index, record in enumerate(
                evaluation_samples,
                start=1,
            ):
                text = str(record["text"])
                true_id = int(record["label"])

                # Select examples for the current test text
                if strategy == "zero_shot":
                    prompt_examples = None

                elif strategy == "good_few_shot":
                    prompt_examples = select_examples_for_text(
                        text=text,
                        candidates=candidates,
                        vectorizer=vectorizer,
                        candidate_matrix=candidate_matrix,
                        class_labels=class_labels,
                        examples_per_class=examples_per_class,
                        most_relevant=True,
                    )

                else:
                    # Poor few-shot: lowest lexical similarity,
                    # but every demonstration keeps its true label.
                    prompt_examples = select_examples_for_text(
                        text=text,
                        candidates=candidates,
                        vectorizer=vectorizer,
                        candidate_matrix=candidate_matrix,
                        class_labels=class_labels,
                        examples_per_class=examples_per_class,
                        most_relevant=False,
                    )

                # Construct prompt
                prompt = build_prompt(
                    dataset_name=dataset_name,
                    text=text,
                    strategy=strategy,
                    examples=prompt_examples,
                )

                # Debug: compare the first IMDb prompt for each strategy
                if dataset_name == "IMDb" and index == 1:
                    print(f"\nStrategy: {strategy}")
                    print("Demonstration examples:", prompt_examples)
                    print("Generated prompt:", prompt)

                # Get model prediction from local Ollama
                raw_response, response_time = generate_response(prompt)

                predicted_id = parse_prediction(
                    raw_response,
                    dataset_name,
                )

                valid_prediction = predicted_id is not None

                # Use -1 for invalid outputs so they count as incorrect
                metric_prediction = (
                    predicted_id if valid_prediction else -1
                )

                y_true.append(true_id)
                y_pred.append(metric_prediction)

                # Save individual prediction details
                all_details.append({
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "examples_per_class": actual_examples_per_class,
                    "sample_number": index,
                    "text": text,
                    "true_label_id": true_id,
                    "true_label": class_labels[true_id],
                    "predicted_label_id": predicted_id,
                    "predicted_label": (
                        class_labels[predicted_id]
                        if valid_prediction
                        else "Invalid"
                    ),
                    "correct": (
                        valid_prediction
                        and predicted_id == true_id
                    ),
                    "valid_prediction": valid_prediction,
                    "raw_response": raw_response,
                    "prompt": prompt,
                    "demonstration_examples": "\n\n".join(
                        f"Text: {str(example['text'])[:700]}\nCategory: {class_labels[int(example['label'])]}"
                        for example in (prompt_examples or [])
                    ),
                    "response_time_seconds": round(
                        response_time,
                        3,
                    ),
                })

                predicted_label = (
                    class_labels[predicted_id]
                    if valid_prediction
                    else "Invalid"
                )

                print(
                    f"  Sample {index}/{len(evaluation_samples)}: "
                    f"true={class_labels[true_id]}, "
                    f"predicted={predicted_label}"
                )

            # Calculate metrics
            metrics = calculate_metrics(
                y_true,
                y_pred,
                class_labels,
            )

            all_summary.append({
                "dataset": dataset_name,
                "strategy": strategy,
                "sample_size": len(evaluation_samples),
                "examples_per_class": actual_examples_per_class,
                "random_seed": random_seed,
                **metrics,
            })

            print(
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro-Precision: {metrics['macro_precision']:.4f} | "
                f"Macro-Recall: {metrics['macro_recall']:.4f} | "
                f"Macro-F1: {metrics['macro_f1']:.4f}"
            )

    # Convert results to DataFrames
    summary_df = pd.DataFrame(all_summary)
    details_df = pd.DataFrame(all_details)

    return summary_df, details_df


def evaluate_custom_prompt(
    dataset_name,
    custom_instructions,
    strategy="zero_shot",
    sample_size=20,
    random_seed=42,
    examples_per_class=1,
):
    """Evaluate custom instructions with zero-shot or either few-shot strategy.

    Demonstrations are selected only from the training split. All strategies
    evaluate the same reproducible test sample when called with the same seed.
    """
    if not str(custom_instructions or "").strip():
        raise ValueError("Please enter custom prompt instructions.")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    train_records, test_records, class_labels = load_classification_dataset(dataset_name)
    evaluation_samples = get_evaluation_samples(
        test_records, sample_size=sample_size, random_seed=random_seed
    )

    candidates = vectorizer = candidate_matrix = None
    if strategy in ("good_few_shot", "poor_few_shot"):
        candidates, vectorizer, candidate_matrix = prepare_example_candidates(
            train_records=train_records,
            class_labels=class_labels,
            examples_per_class=examples_per_class,
            random_seed=random_seed,
            candidates_per_class=100,
        )

    y_true, y_pred, details = [], [], []
    for index, record in enumerate(evaluation_samples, start=1):
        text = str(record["text"])
        true_id = int(record["label"])
        prompt_examples = None
        if strategy in ("good_few_shot", "poor_few_shot"):
            prompt_examples = select_examples_for_text(
                text=text,
                candidates=candidates,
                vectorizer=vectorizer,
                candidate_matrix=candidate_matrix,
                class_labels=class_labels,
                examples_per_class=examples_per_class,
                most_relevant=(strategy == "good_few_shot"),
            )

        prompt = build_custom_prompt(
            dataset_name=dataset_name,
            text=text,
            custom_instructions=custom_instructions,
            strategy=strategy,
            examples=prompt_examples,
        )
        raw_response, response_time = generate_response(prompt)
        predicted_id = parse_prediction(raw_response, dataset_name)
        valid_prediction = predicted_id is not None
        y_true.append(true_id)
        y_pred.append(predicted_id if valid_prediction else -1)

        details.append({
            "dataset": dataset_name,
            "strategy": f"custom_{strategy}",
            "sample_number": index,
            "text": text,
            "true_label_id": true_id,
            "true_label": class_labels[true_id],
            "predicted_label_id": predicted_id,
            "predicted_label": class_labels[predicted_id] if valid_prediction else "Invalid",
            "correct": valid_prediction and predicted_id == true_id,
            "valid_prediction": valid_prediction,
            "raw_response": raw_response,
            "prompt": prompt,
            "demonstration_examples": "\n\n".join(
                f"Text: {str(example['text'])[:700]}\nCategory: {class_labels[int(example['label'])]}"
                for example in (prompt_examples or [])
            ),
            "response_time_seconds": round(response_time, 3),
        })

    metrics = calculate_metrics(y_true, y_pred, class_labels)
    summary = pd.DataFrame([{
        "dataset": dataset_name,
        "strategy": f"custom_{strategy}",
        "sample_size": len(evaluation_samples),
        "examples_per_class": examples_per_class if strategy != "zero_shot" else 0,
        "random_seed": random_seed,
        **metrics,
    }])
    return summary, pd.DataFrame(details)


def save_results(summary_df, details_df):
    """
    Save timestamped results and update latest CSV files.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    details_path = RESULTS_DIR / f"details_{timestamp}.csv"

    # Save timestamped files
    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    # Update latest result files
    summary_df.to_csv(
        RESULTS_DIR / "latest_summary.csv",
        index=False,
    )

    details_df.to_csv(
        RESULTS_DIR / "latest_details.csv",
        index=False,
    )

    print("\nResults saved successfully:")
    print("Summary:", summary_path)
    print("Details:", details_path)
    print("Latest summary:", RESULTS_DIR / "latest_summary.csv")
    print("Latest details:", RESULTS_DIR / "latest_details.csv")