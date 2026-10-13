# 🤖 LLM Prompting Strategy Comparison

<p align="center">
  <b>Zero-Shot vs Good Few-Shot vs Poor Few-Shot Prompting</b>
  <br>
  Customer Complaint Classification using a Local Large Language Model
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--3B-purple" alt="Qwen2.5-3B">
  <img src="https://img.shields.io/badge/Runtime-Ollama-black?logo=ollama" alt="Ollama">
  <img src="https://img.shields.io/badge/Task-Text%20Classification-teal" alt="Text Classification">
  <img src="https://img.shields.io/badge/Status-Academic%20Project-success" alt="Academic Project">
</p>

---

## 📌 1. Project Overview

This project investigates how different prompting strategies affect the performance of a **Large Language Model (LLM)** on a text-classification task.

The selected task is **customer complaint classification**, where an LLM categorizes customer complaints into four categories: Billing, Technical, Delivery, and Account.

The experiment compares three prompting strategies:

| Strategy | Description |
|---|---|
| 🎯 **Zero-Shot** | Classifies complaints using instructions without labeled examples. |
| ✅ **Good Few-Shot** | Uses correctly labeled examples in the prompt. |
| ⚠️ **Poor Few-Shot** | Uses deliberately incorrect labels to study the effect of misleading examples. |

The project uses **Qwen2.5:3B** through **Ollama**, allowing the model to run locally without requiring an OpenAI API key.

## 🎯 2. Problem Statement

Customer-support teams receive many complaints related to payments, software errors, deliveries, and account access. Manually sorting these complaints can be time-consuming and inconsistent.

An automated classification system can help route complaints to the appropriate support team. However, an LLM's performance may depend on how the task is described and whether the prompt contains examples.

This project investigates whether examples improve classification performance and whether incorrect demonstrations can confuse the model.

## 🔬 3. Research Question & Hypotheses

### ❓ Research Question

How do zero-shot, good few-shot, and poor few-shot prompting strategies affect the performance of an LLM on customer complaint classification?

### 💡 Hypotheses

- **H1:** Correctly labeled examples may improve classification performance.
- **H2:** Incorrectly labeled examples may reduce classification performance.
- **H3:** Clear instructions may allow zero-shot prompting to perform competitively with few-shot prompting.

These hypotheses are tested experimentally rather than assumed to be true.

## 📂 4. Dataset Description

The project uses a balanced dataset of **1,000 synthetic, template-generated customer complaints**.

| 🏷️ Category | 📝 Description | 📊 Examples |
|---|---|---:|
| 💳 Billing | Payments, charges, invoices, and refunds | 250 |
| 🛠️ Technical | Software bugs, application crashes, and website errors | 250 |
| 📦 Delivery | Shipping, tracking, and parcel-delivery issues | 250 |
| 🔐 Account | Login, passwords, profiles, and account access | 250 |
| **Total** | | **1,000** |

### 📊 Train-Test Split

The dataset is divided using a stratified 75:25 split.

- 🧠 **Training set:** 750 examples
- 🧪 **Testing set:** 250 examples

The training set supplies examples for few-shot prompts. The same test set is used to evaluate all three prompting strategies.

> ⚠️ **Dataset limitation:** The dataset is synthetic and template-generated, not a collection of 1,000 independently collected real-world complaints. Similar wording may appear in both training and testing data, potentially inflating evaluation scores.

## ⚙️ 5. Methodology

The same language model is evaluated under three prompting conditions.

### 🎯 A. Zero-Shot Prompting

The model receives task instructions, category definitions, and a new complaint. No labeled examples are provided.

**Objective:** Establish a baseline for classification without demonstrations.

### ✅ B. Good Few-Shot Prompting

The prompt contains correctly labeled complaint examples selected from the training set.

**Objective:** Evaluate whether relevant examples help the model understand the expected relationship between complaint text and category labels.

### ⚠️ C. Poor Few-Shot Prompting

The prompt contains complaint examples paired with deliberately incorrect category labels.

**Objective:** Investigate whether misleading examples can confuse the model and reduce classification performance.

### ⚖️ Experimental Fairness

The model, test set, category definitions, output format, and evaluation metrics remain consistent across the three conditions. The prompting strategy is the primary variable being compared.

**Important:** Few-shot prompting does not fine-tune or retrain the model. The examples are provided in the prompt context at inference time.

## 🧰 6. Technology Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Main programming language |
| 🧠 Qwen2.5:3B | Local language model |
| 🦙 Ollama | Running the model locally |
| 🐼 Pandas | Dataset processing |
| 📏 Scikit-learn | Evaluation metrics |
| 📈 Matplotlib | Charts and confusion matrices |
| 📝 CSV / JSON | Storing predictions and results |

## 🗂️ 7. Project Structure

```text
llm-prompting-comparison/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 🐍 main.py
├── 🐍 dataset.py
├── 🐍 prompts.py
├── 🐍 evaluate.py
│
├── 📁 data/
│   └── complaints.csv
│
└── 📁 results/
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

### 📌 Main Files

- `main.py` — Runs the complete experiment and generates results and visualizations.
- `dataset.py` — Generates the complaint dataset and creates training and testing splits.
- `prompts.py` — Defines category instructions and constructs the three prompting strategies.
- `evaluate.py` — Communicates with Ollama, extracts predictions, and calculates metrics.
- `requirements.txt` — Lists the required Python packages.
- `results/` — Stores predictions, metrics, charts, and confusion matrices.

The dataset and result files are generated when the program runs.

## 💻 8. Installation & Setup

### 📋 Prerequisites

Install the following before running the project:

- Python 3.10 or later recommended
- Ollama
- Qwen2.5:3B model
- Required Python packages

🔗 **Install Ollama:** https://ollama.com/download

### Step 1️⃣ — Download the Model

Open a terminal and run:

```bash
ollama pull qwen2.5:3b
```

Verify the installed model:

```bash
ollama list
```

### Step 2️⃣ — Create a Virtual Environment

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

### Step 3️⃣ — Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Your `requirements.txt` should contain:

```text
ollama
pandas
scikit-learn
matplotlib
```

### Step 4️⃣ — Check the Model Configuration

Open `evaluate.py` and confirm:

```python
MODEL_NAME = "qwen2.5:3b"
```

If you use a different model, update the name to match the output of `ollama list`.

## ▶️ 9. Running the Project

Run the following command from the project root directory:

```bash
python main.py
```

The program will:

1. Generate the complaint dataset.
2. Create stratified training and testing splits.
3. Build prompts for all three strategies.
4. Classify the test complaints using the local LLM.
5. Calculate Accuracy, Precision, Recall, and Macro-F1.
6. Save predictions and detailed evaluation results.
7. Generate a comparison chart and confusion matrices.

⏳ The first run may take some time because the model classifies test examples under each prompting strategy.

## 📏 10. Evaluation Metrics

The experiment uses four standard classification metrics.

| Metric | Meaning |
|---|---|
| 🎯 **Accuracy** | Proportion of all predictions that are correct. |
| 🔍 **Precision** | How many predictions for a category are actually correct. |
| 📥 **Recall** | How many actual examples of a category are correctly identified. |
| ⚖️ **Macro-F1** | Average of the F1-scores across all categories, giving each category equal weight. |

**Accuracy formula:**

`Accuracy = Correct Predictions / Total Predictions`

Macro-averaged metrics help evaluate performance across categories without giving a more frequent category extra weight.

## 📈 11. Experimental Results

The following metrics are calculated from the confusion matrices supplied for the experiment.

| Prompting Strategy | Correct Predictions | Accuracy | Macro Precision | Macro Recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 🎯 Zero-Shot | 246/250 | 98.40% | 98.40% | 98.40% | 98.40% |
| ✅ Good Few-Shot | 243/250 | 97.20% | 97.38% | 97.20% | 97.20% |
| ⚠️ Poor Few-Shot | 232/250 | 92.80% | 94.06% | 92.80% | 92.86% |

> 📌 **Note:** These values are calculated from the supplied confusion matrices. If the code generates different results when rerun, update this table using the latest output files.

### 🔍 Results Interpretation

- 🥇 **Zero-shot performed best**, achieving 98.40% accuracy.
- 🥈 **Good few-shot achieved 97.20% accuracy**, slightly below zero-shot.
- ⚠️ **Poor few-shot achieved 92.80% accuracy**, the lowest of the three strategies.
- Incorrect demonstrations were associated with more classification errors.

The results suggest that examples do not automatically improve model performance. Their relevance and correctness matter.

However, this experiment does not prove that zero-shot prompting is always better. Results depend on the model, dataset, prompt construction, and evaluation method.

## 🧩 12. Confusion Matrix Analysis

A confusion matrix compares actual categories with predicted categories.

- **Rows:** Actual categories
- **Columns:** Predicted categories
- **Diagonal cells:** Correct predictions
- **Off-diagonal cells:** Incorrect predictions

In the supplied poor few-shot confusion matrix, **12 actual Account complaints were predicted as Billing**.

This pattern is consistent with the possibility that misleading examples affected classification decisions. Further examination of individual prompts and predictions would be needed to determine the precise cause.

## ⚠️ 13. Limitations

1. 🧪 **Synthetic dataset:** The complaints are template-generated rather than independently collected from real customer-support interactions.
2. 🔁 **Potential data leakage:** Similar wording may occur in both the training and testing sets.
3. 🧠 **Single model:** The reported results represent one model configuration.
4. 🎲 **Single split:** The experiment does not measure variability across multiple random seeds.
5. 🌍 **Limited generalization:** Performance on template-generated complaints may not represent real-world performance on ambiguous or complex complaints.

## 🚀 14. Future Improvements

Possible improvements include:

- Use real, anonymized customer complaints or independently authored examples.
- Split base complaints before generating wording variations to reduce near-duplicate leakage.
- Compare multiple LLMs and model sizes.
- Repeat experiments with multiple random seeds and report mean and standard deviation.
- Examine per-category precision, recall, and F1-score.
- Investigate individual failure cases and ambiguous complaints.
- Test different numbers and orders of few-shot examples.
- Record model versions and system configurations for reproducibility.

## 🛠️ 15. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'prompts'`

1. Ensure `prompts.py` is in the same directory as `evaluate.py`.
2. Confirm the filename is exactly `prompts.py`, not `prompts.py.txt`.
3. Open the project root directory in your editor.
4. Run this command from the project root:

```bash
python -c "from prompts import CATEGORIES, build_prompt; print(CATEGORIES)"
```

5. In Visual Studio Code, select the correct Python interpreter and reload the window if necessary.

### ❌ `'ollama' is not recognized`

Confirm that Ollama is installed, open a new terminal, and run:

```bash
ollama --version
```

If the command remains unavailable, check the installation and PATH configuration.

### ❌ Model Not Found

Run:

```bash
ollama list
ollama pull qwen2.5:3b
```

Ensure `MODEL_NAME` in `evaluate.py` matches the installed model.

### ⚠️ Suspiciously High or Identical Scores

If all prompting strategies produce 100% or nearly identical scores:

- Inspect `results/predictions.csv`.
- Verify that test labels are not included in the prompt examples.
- Check for similar or duplicate examples across the training and testing splits.
- Confirm that evaluation uses actual model predictions.
- Test the model on independently written complaints.
- Repeat the evaluation with additional train/test splits.

## 🏁 16. Conclusion

This project compares zero-shot, good few-shot, and deliberately poor few-shot prompting for customer complaint classification using a locally running LLM.

In the supplied results, zero-shot achieved the highest performance, while poor few-shot prompting produced the lowest performance. The experiment demonstrates that prompt design and example quality can influence classification outcomes.

Because the dataset is synthetic and template-based, additional evaluation on realistic, independent data is required before drawing conclusions about real-world customer-support systems.


  <br>
  Built as an academic project to study LLM prompting and text classification.
</p>
