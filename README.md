# LLM Prompting Strategy Comparison for Text Classification

## 1. Project Overview

This project investigates how different prompting strategies affect the performance of a Large Language Model (LLM) on a text-classification task.

The selected task is **customer complaint classification**, in which a language model assigns each complaint to one of four categories: Billing, Technical, Delivery, or Account.

The project compares three prompting strategies:

1. **Zero-Shot Prompting:** The model classifies a complaint using task instructions and category definitions without receiving labeled examples.
2. **Good Few-Shot Prompting:** The model receives correctly labeled examples in the prompt before classifying a new complaint.
3. **Poor Few-Shot Prompting:** The model receives examples with deliberately incorrect labels to investigate how misleading demonstrations affect classification performance.

The experiment uses the **Qwen2.5:3B** language model through **Ollama**, allowing the model to run locally without requiring an OpenAI API key.

## 2. Problem Statement

Customer-support departments receive a large number of complaints about payments, software errors, deliveries, and account access. Manually categorizing every complaint can be time-consuming and inconsistent.

An automated classification system can help route complaints to the appropriate support team. However, its performance may depend on how the task is described and whether examples are included in the prompt.

This project evaluates whether correctly labeled examples improve classification performance and whether incorrectly labeled examples negatively affect the model's predictions.

## 3. Research Question

**How do zero-shot, good few-shot, and deliberately poor few-shot prompting strategies affect the performance of an LLM on customer complaint classification?**

### Research Hypotheses

- **H1:** Correctly labeled examples may improve classification consistency.
- **H2:** Incorrectly labeled examples may reduce classification performance.
- **H3:** Clear instructions and category definitions may allow zero-shot prompting to perform competitively with few-shot prompting.

These hypotheses are evaluated using classification metrics rather than assumed to be true in advance.

## 4. Dataset Description

The project uses a balanced dataset of **1,000 template-generated customer complaints**.

| Category | Description | Number of Examples |
|---|---|---:|
| Billing | Payments, charges, invoices, and refunds | 250 |
| Technical | Software bugs, application crashes, and website errors | 250 |
| Delivery | Shipping, tracking, and parcel-delivery issues | 250 |
| Account | Login, passwords, profiles, and account access | 250 |
| **Total** | | **1,000** |

The dataset is generated programmatically using base complaint descriptions and wording patterns.

### Train-Test Split

The dataset is divided using a stratified 75:25 split.

- **Training set:** 750 examples
- **Testing set:** 250 examples

The training data provides examples for few-shot prompting. The test set is used to evaluate all three prompting strategies.

**Dataset limitation:** The dataset is synthetic and template-generated. Similar wording may appear in both the training and testing sets, potentially inflating the evaluation scores. The results should therefore be interpreted as a controlled demonstration rather than proof of real-world performance.

## 5. Methodology

The same local language model is evaluated under three prompting conditions.

### 5.1 Zero-Shot Prompting

The model receives the task instructions, category definitions, and a new complaint.

No labeled examples are provided.

**Purpose:** Establish a baseline for classification without demonstrations.

### 5.2 Good Few-Shot Prompting

The prompt contains correctly labeled examples selected from the training set.

**Purpose:** Determine whether relevant examples help the model understand the expected relationship between complaint text and category labels.

### 5.3 Poor Few-Shot Prompting

The prompt contains complaint examples paired with deliberately incorrect category labels.

**Purpose:** Investigate whether misleading examples can confuse the model and reduce classification performance.

### Experimental Fairness

The model, test set, category definitions, output format, and evaluation metrics are kept consistent across the three conditions. The prompting strategy is the primary variable being compared.

Few-shot prompting does not fine-tune or update the model's parameters. Instead, the examples are included in the prompt context.

## 6. Technology Stack

- **Programming language:** Python
- **Language model:** Qwen2.5:3B
- **Local model runtime:** Ollama
- **Data processing:** Pandas
- **Evaluation metrics:** Scikit-learn
- **Visualization:** Matplotlib
- **Output format:** CSV, JSON, and PNG
- **Development environment:** Visual Studio Code or another Python-compatible editor

No OpenAI API key is required.

## 7. Project Structure

```text
llm-prompting-comparison/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── main.py
├── dataset.py
├── prompts.py
├── evaluate.py
│
├── data/
│   └── complaints.csv
│
└── results/
    ├── train_data.csv
    ├── test_data.csv
    ├── metrics.csv
    ├── predictions.csv
    ├── detailed_results.json
    ├── comparison_chart.png
    ├── zero_shot_confusion_matrix.png
    ├── good_few_shot_confusion_matrix.png
    └── poor_few_shot_confusion_matrix.png
```

The dataset and result files are generated when the experiment runs. They may not exist in a fresh checkout until execution.

### Main Files

- `main.py`: Runs the complete experiment and generates result files and visualizations.
- `dataset.py`: Generates the complaint dataset and creates the training and testing splits.
- `prompts.py`: Defines the category instructions and constructs the three prompting strategies.
- `evaluate.py`: Sends prompts to Ollama, extracts predicted labels, and calculates evaluation metrics.
- `requirements.txt`: Lists the required Python packages.
- `results/`: Stores evaluation results, predictions, charts, and confusion matrices.

## 8. Installation and Setup

### Prerequisites

Install Python and Ollama before running the experiment.

Ollama download page: https://ollama.com/download

### Step 1: Download the Language Model

Open a terminal and run:

```bash
ollama pull qwen2.5:3b
```

Verify that the model is installed:

```bash
ollama list
```

The model name configured in `evaluate.py` must match the installed model name.

### Step 2: Create a Virtual Environment

Run these commands from the project root directory.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
ollama
pandas
scikit-learn
matplotlib
```

### Step 4: Verify the Model Name

Open `evaluate.py` and check:

```python
MODEL_NAME = "qwen2.5:3b"
```

If a different model is installed, update the name accordingly.

## 9. Running the Project

Ensure that Ollama is installed and the required model is available.

Open a terminal in the project root directory and run:

```bash
python main.py
```

The program will:

1. Generate the complaint dataset.
2. Create stratified training and testing splits.
3. Construct zero-shot, good few-shot, and poor few-shot prompts.
4. Classify the test complaints using the local model.
5. Calculate Accuracy, Precision, Recall, and Macro-F1.
6. Save individual predictions and detailed evaluation results.
7. Generate a comparison chart and confusion matrices.

The first run may take some time because the model processes test examples under all three prompting strategies.

## 10. Evaluation Metrics

The experiment uses four standard classification metrics.

### Accuracy

The proportion of all predictions that are correct.

**Formula:**

Accuracy = Correct Predictions / Total Predictions

### Precision

Precision measures how many examples predicted as a particular category actually belong to that category.

### Recall

Recall measures how many examples belonging to a particular category are correctly identified.

### Macro-F1

Macro-F1 calculates the F1-score for each category and then takes the unweighted average across all categories.

Macro-averaged metrics give each category equal importance.

## 11. Experimental Results

The following results are calculated from the confusion matrices supplied for the experiment.

| Prompting Strategy | Correct Predictions | Accuracy | Macro Precision | Macro Recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Zero-Shot | 246/250 | 98.40% | 98.40% | 98.40% | 98.40% |
| Good Few-Shot | 243/250 | 97.20% | 97.38% | 97.20% | 97.20% |
| Poor Few-Shot | 232/250 | 92.80% | 94.06% | 92.80% | 92.86% |

*Note: These figures are calculated from the supplied confusion matrices. If the experiment is rerun and produces different predictions, use the newly generated results in the final report.*

### Interpretation

- **Zero-shot performed best** in the supplied results, with 98.40% accuracy.
- **Good few-shot achieved 97.20% accuracy**, slightly below zero-shot.
- **Poor few-shot achieved 92.80% accuracy**, the lowest of the three strategies.
- Incorrect demonstrations were associated with more classification errors.

These findings suggest that providing examples does not automatically improve performance. The quality and relevance of the examples matter.

However, this experiment does not establish that zero-shot prompting is always superior. The outcome depends on the model, dataset, prompt construction, and evaluation method.

## 12. Confusion Matrix Analysis

A confusion matrix compares the actual categories with the predicted categories.

- **Rows:** Actual categories
- **Columns:** Predicted categories
- **Diagonal cells:** Correct predictions
- **Off-diagonal cells:** Incorrect predictions

In the supplied poor few-shot confusion matrix, 12 actual Account complaints were predicted as Billing.

This pattern is consistent with the possibility that misleading examples affected the model's category decisions. The confusion matrix shows where the mistakes occurred, but further analysis of individual prompts and predictions would be needed to establish the exact reason.

## 13. Limitations

The project has several limitations:

1. **Synthetic data:** The complaints are generated from templates rather than collected independently from real customer-support interactions.
2. **Potential data leakage through similar wording:** Related sentences may appear in both training and test sets.
3. **Single model:** Only one model configuration is represented in the reported results.
4. **Single split:** The reported experiment does not measure variation across multiple random seeds.
5. **Limited generalization:** Results on template-generated complaints may not represent performance on ambiguous, informal, or complex real-world complaints.

## 14. Future Improvements

Future versions of this project could:

- Use real, anonymized customer complaints or independently authored examples.
- Split the base complaints before generating wording variations to prevent near-duplicate leakage.
- Evaluate multiple LLMs and model sizes.
- Repeat experiments with multiple random seeds and report mean and standard deviation.
- Examine per-category precision, recall, and F1-score.
- Investigate individual failure cases and ambiguous complaints.
- Test different numbers, ordering, and selection methods for few-shot examples.

## 15. Troubleshooting

### Error: `ModuleNotFoundError: No module named 'prompts'`

Check the following:

1. `prompts.py` is in the same directory as `evaluate.py`.
2. The filename is exactly `prompts.py`, not `prompts.py.txt`.
3. The terminal is opened in the project root directory.
4. The correct Python interpreter or virtual environment is selected.

Test the import from the project root:

```bash
python -c "from prompts import CATEGORIES, build_prompt; print(CATEGORIES)"
```

### Error: `'ollama' is not recognized`

Confirm that Ollama is installed, open a new terminal, and run:

```bash
ollama --version
```

If the command remains unavailable, check the installation and PATH configuration.

### Error: Model not found

Run:

```bash
ollama list
```

If the model is missing, download it:

```bash
ollama pull qwen2.5:3b
```

Ensure the configured model name matches the installed model.

### Suspiciously High or Identical Scores

If all prompting strategies return 100% or nearly identical metrics:

- Inspect `results/predictions.csv`.
- Verify that the test labels are not being included as demonstrations.
- Check whether near-duplicate examples appear across the training and testing splits.
- Confirm that the evaluation code uses actual model predictions rather than expected labels.
- Evaluate on independently written complaints and use more than one train/test split.

## 16. Conclusion

This project presents a controlled comparison of zero-shot, good few-shot, and poor few-shot prompting for customer complaint classification using a local LLM.

In the supplied results, zero-shot achieved the highest performance, while deliberately incorrect examples resulted in the lowest performance. The experiment demonstrates that prompt design and example quality can influence classification outcomes.

Because the dataset is synthetic and template-based, further evaluation on realistic, independent data is necessary before drawing conclusions about real-world customer-support systems.


