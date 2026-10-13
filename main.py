
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dataset import prepare_data
from prompts import CATEGORIES
from evaluate import evaluate_strategy, MODEL_NAME


def main():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and fixed train/test split.
    train_df, test_df = prepare_data()

    print("=" * 65)
    print("LLM PROMPTING STRATEGY COMPARISON")
    print("=" * 65)
    print(f"Model: {MODEL_NAME}")
    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples: {len(test_df)}")
    print(f"Categories: {', '.join(CATEGORIES)}")

    strategies = [
        "zero_shot",
        "good_few_shot",
        "poor_few_shot"
    ]

    all_metrics = []
    all_predictions = []
    detailed_results = {}

    # Evaluate every strategy on the same test data.
    for strategy in strategies:
        print(f"\nRunning strategy: {strategy}")
        print("-" * 65)

        metrics, predictions, report, matrix = (
            evaluate_strategy(train_df, test_df, strategy)
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)

        detailed_results[strategy] = {
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "categories": CATEGORIES
        }

    # Save metrics and predictions.
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        results_dir / "metrics.csv",
        index=False
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )
    predictions_df.to_csv(
        results_dir / "predictions.csv",
        index=False
    )

    with open(
        results_dir / "detailed_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(detailed_results, file, indent=4)

    # Display final metric comparison.
    print("\nFINAL COMPARISON")
    print("=" * 90)
    print(
        metrics_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.3f}".format,
                "Precision": "{:.3f}".format,
                "Recall": "{:.3f}".format,
                "Macro-F1": "{:.3f}".format
            }
        )
    )

    # Generate the comparison chart.
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "Macro-F1"
    ]

    ax = metrics_df.set_index("Strategy")[
        metric_names
    ].plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
        rot=0
    )

    ax.set_title("LLM Prompting Strategy Comparison")
    ax.set_xlabel("Prompting Strategy")
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(
        results_dir / "comparison_chart.png",
        dpi=200
    )
    plt.close()

    # Generate a confusion matrix for each strategy.
    for strategy in strategies:
        matrix = detailed_results[strategy][
            "confusion_matrix"
        ]

        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(matrix, cmap="Blues")

        ax.set_title(f"Confusion Matrix: {strategy}")
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("Actual category")
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CATEGORIES, rotation=30)
        ax.set_yticklabels(CATEGORIES)

        for i in range(len(CATEGORIES)):
            for j in range(len(CATEGORIES)):
                ax.text(
                    j,
                    i,
                    matrix[i][j],
                    ha="center",
                    va="center"
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        fig.savefig(
            results_dir / f"{strategy}_confusion_matrix.png",
            dpi=200
        )
        plt.close(fig)

    print("\nExperiment completed successfully.")
    print("All results have been saved in the results/ folder.")


if __name__ == "__main__":
    main()
