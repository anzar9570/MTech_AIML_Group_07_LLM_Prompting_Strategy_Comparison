
from src.datasets_loader import load_classification_dataset
from src.datasets_loader import get_evaluation_samples


for dataset_name in ["IMDb", "AG News"]:
    print(f"\n===== {dataset_name} =====")

    train_records, test_records, class_labels = (
        load_classification_dataset(dataset_name)
    )

    samples = get_evaluation_samples(
        test_records,
        sample_size=20,
        random_seed=42,
    )

    print("Training examples:", len(train_records))
    print("Testing examples:", len(test_records))
    print("Class labels:", class_labels)
    print("Evaluation sample size:", len(samples))

    print("\nFirst two evaluation samples:")
    for sample in samples[:2]:
        print("Text:", sample["text"][:150])
        print("Label:", class_labels[sample["label"]])
        print()
