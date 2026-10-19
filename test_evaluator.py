from src.evaluator import run_evaluation, save_results


if __name__ == "__main__":
    summary_df, details_df = run_evaluation(
        dataset_names=["IMDb", "AG News"],
        sample_size=20,
        random_seed=42,
        examples_per_class=1,
    )

    print("\n===== FINAL RESULTS =====")
    print(summary_df.to_string(index=False))

    save_results(summary_df, details_df)