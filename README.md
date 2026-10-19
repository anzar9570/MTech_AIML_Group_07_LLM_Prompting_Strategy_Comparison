
<div align="center">

# 🧠 PromptEval
### Evaluating Zero-Shot and Few-Shot Prompting Strategies for LLM-Based Text Classification

**A comparative experimental study using IMDb and AG News datasets with a locally deployed Large Language Model.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow?logo=huggingface)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/Project-Academic-green)

**🎓 M.Tech — Computer Science & Engineering**

**Birla Institute of Technology, Mesra, Ranchi**

</div>

---

## 📌 Project Overview

**PromptEval** is an interactive LLM evaluation system designed to compare different prompting strategies for text classification.

The project investigates how the quality and relevance of few-shot examples influence the classification performance of a Large Language Model (LLM).

Three prompting strategies are evaluated:

- 🎯 **Zero-Shot Prompting:** Classification without demonstration examples.
- ✅ **Good Few-Shot Prompting:** Classification using highly relevant labelled examples.
- ⚠️ **Poor Few-Shot Prompting:** Classification using less-relevant but correctly labelled examples.

The system uses a locally deployed **Qwen2.5:3b** model through Ollama and provides an interactive Streamlit dashboard for evaluation and visualization.

---

## 🌍 Real-World Problem

Text classification is widely used in real-world applications such as:

- Customer review sentiment analysis
- News article categorization
- Customer support ticket classification
- Social media monitoring
- Automated document organization

Traditional machine learning approaches often require task-specific training.

Large Language Models can perform classification through prompting without additional model training.

However, classification performance may depend on the quality of the examples included in the prompt.

**This project investigates whether carefully selected examples improve LLM classification performance compared with less-relevant examples and zero-shot prompting.**

---

## 🎯 Research Question

> How does the relevance of few-shot demonstration examples affect the classification performance of a Large Language Model across different text classification datasets?

### Research Hypothesis

**H1:** Good Few-Shot prompting will generally outperform Zero-Shot prompting.

**H2:** Relevant few-shot examples will generally produce better classification performance than less-relevant examples.

**H3:** The effect of demonstration relevance may differ across datasets and classification tasks.

These hypotheses are evaluated experimentally rather than assumed to be true.

---

## 📚 Datasets

Two publicly available datasets from Hugging Face are used.

### 🎬 1. IMDb Dataset

**Task:** Binary sentiment classification

| Property | Description |
|---|---|
| Dataset | IMDb |
| Source | Hugging Face |
| Training samples | 25,000 |
| Testing samples | 25,000 |
| Number of classes | 2 |
| Labels | Negative, Positive |

**Dataset:** https://huggingface.co/datasets/stanfordnlp/imdb

### 📰 2. AG News Dataset

**Task:** Multi-class news classification

| Property | Description |
|---|---|
| Dataset | AG News |
| Source | Hugging Face |
| Training samples | 120,000 |
| Testing samples | 7,600 |
| Number of classes | 4 |
| Labels | World, Sports, Business, Sci/Tech |

**Dataset:** https://huggingface.co/datasets/fancyzhx/ag_news

### Dataset Usage

- Training data is used only to select few-shot demonstrations.
- Test data is used for model evaluation.
- The same test samples are evaluated across all prompting strategies.
- A fixed random seed supports reproducible sample selection.

---

## 🧪 Prompting Strategies

### 1️⃣ Zero-Shot Prompting

The model receives task instructions and the input text without any labelled examples.

**Example:**

```text
Classify the following movie review.

Allowed categories: Positive, Negative

Text: The movie was absolutely fantastic.

Category:
```

### 2️⃣ Good Few-Shot Prompting

The model receives correctly labelled training examples selected based on high lexical similarity to the input text.

**Example:**

```text
Classify the following movie review.

Allowed categories: Positive, Negative

Text: The acting was excellent.
Category: Positive

Text: The movie was boring.
Category: Negative

Now classify:

Text: The performances were outstanding.

Category:
```

### 3️⃣ Poor Few-Shot Prompting

The model receives correctly labelled examples with relatively low lexical similarity to the input text.

Unlike Good Few-Shot prompting, the examples are deliberately selected to be less relevant.

**Important:** Poor Few-Shot examples are not intentionally mislabelled.

This allows the experiment to study the effect of example relevance while preserving label correctness.

---

## ⚙️ Methodology

The system follows these steps:

```text
                START
                  |
                  v
          Select Dataset
          (IMDb / AG News)
                  |
                  v
        Load Training & Test Data
                  |
                  v
       Select Evaluation Samples
                  |
                  v
        Choose Prompting Strategy
                  |
        +---------+---------+
        |         |         |
        v         v         v
     Zero-Shot   Good      Poor
                 Few-Shot  Few-Shot
        |         |         |
        |         v         v
        |      Highest     Lowest
        |      TF-IDF      TF-IDF
        |      Similarity  Similarity
        |         |         |
        +---------+---------+
                  |
                  v
          Construct Prompt
                  |
                  v
         Local Ollama Model
            (Qwen2.5:3b)
                  |
                  v
         Parse Predicted Label
                  |
                  v
         Calculate Metrics
                  |
                  v
         Visualize and Save
                Results
```

### 🔍 TF-IDF and Cosine Similarity

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to represent text numerically.

Cosine similarity measures the lexical similarity between the input text and candidate training examples.


### 🔍 Cosine Similarity

$$
\text{Cosine Similarity}(A,B) = \frac{A \cdot B}{\|A\| \|B\|}
$$


**Good Few-Shot:**

Selects examples with the highest cosine similarity within each class.

**Poor Few-Shot:**

Selects examples with the lowest cosine similarity within each class.

This selection is performed separately for each evaluation text.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Ollama | Local LLM execution |
| Qwen2.5:3b | Text classification model |
| Hugging Face Datasets | Dataset loading |
| Scikit-learn | TF-IDF, cosine similarity and evaluation metrics |
| Pandas | Data processing and CSV export |
| Plotly | Interactive result visualization |

---

## 🔬 Experimental Setup

| Parameter | Configuration |
|---|---|
| LLM | Qwen2.5:3b |
| Model execution | Local Ollama |
| Datasets | IMDb, AG News |
| Evaluation samples | 100 per dataset |
| Prompting strategies | 3 |
| Examples per class | 1 for few-shot |
| Zero-Shot examples | 0 |
| Random seed | 42 |
| Candidate pool | Up to 100 training examples per class |
| Total model requests | 600 |

### Evaluation Metrics

The following metrics are calculated:

- **Accuracy:** Percentage of correctly classified samples.
- **Macro-Precision:** Average precision across all classes.
- **Macro-Recall:** Average recall across all classes.
- **Macro-F1:** Average F1-score across all classes.

Macro-averaging gives equal importance to every class.

---

## 📊 Experimental Results

### 🎬 IMDb — Sentiment Classification

| Prompting Strategy | Accuracy |
|---|---:|
| Zero-Shot | 92% |
| Good Few-Shot | 95% |
| Poor Few-Shot | 95% |

**Observation:**

Both Good Few-Shot and Poor Few-Shot achieved identical accuracy on IMDb.

However, further analysis showed that the two strategies did not produce identical predictions for every sample.

### 📰 AG News — News Classification

| Prompting Strategy | Accuracy |
|---|---:|
| Zero-Shot | 68% |
| Good Few-Shot | 78% |
| Poor Few-Shot | 69% |

**Observation:**

Good Few-Shot achieved the highest accuracy on AG News.

It improved accuracy by:

- **10 percentage points** compared with Zero-Shot.
- **9 percentage points** compared with Poor Few-Shot.

These results suggest that relevant demonstrations may be particularly useful for the AG News classification task.

### 📁 Detailed Experimental Results

The complete experimental results are available below:

| File | Description |
|---|---|
| 📈 [latest_summary_.csv](results/latest_summary_.csv) | Evaluation metrics for each dataset and prompting strategy |
| 🔍 [latest_details.csv](results/latest_details.csv) | Individual predictions, prompts, demonstrations and model responses |

**Note:** The results represent a 100-sample experiment, not full-dataset benchmark performance.

---

## 🔎 Interesting Finding: Identical IMDb Accuracy

One interesting observation was that Good Few-Shot and Poor Few-Shot achieved identical accuracy on IMDb.

To investigate this, the individual predictions were compared.

### Prediction Comparison

| Observation | Result |
|---|---:|
| Total IMDb samples | 100 |
| Identical predictions | 96 |
| Different predictions | 4 |
| Good Few-Shot accuracy | 95% |
| Poor Few-Shot accuracy | 95% |
| Identical demonstration sets | 0 |
| Identical generated prompts | 0 |

### Different Prediction Cases

| Sample | Actual Label | Good Few-Shot | Poor Few-Shot |
|---|---|---|---|
| 57 | Positive | Positive ✅ | Negative ❌ |
| 58 | Positive | Positive ✅ | Negative ❌ |
| 61 | Positive | Negative ❌ | Positive ✅ |
| 100 | Positive | Negative ❌ | Positive ✅ |

### Explanation

Good Few-Shot correctly classified samples 57 and 58, whereas Poor Few-Shot classified them incorrectly.

Poor Few-Shot correctly classified samples 61 and 100, whereas Good Few-Shot classified them incorrectly.

Therefore, both strategies achieved the same number of correct predictions.

**Key Finding:**

> Identical evaluation metrics do not necessarily imply identical model predictions or identical prompting behavior.

This demonstrates the importance of examining individual predictions rather than relying only on aggregate metrics.

---

## 🖥️ Streamlit Dashboard Features

PromptEval provides an interactive web interface.

### 📊 Results Module

- Displays evaluation metrics.
- Compares prompting strategies.
- Visualizes classification performance.

### 🔍 Predictions Module

- Displays individual test samples.
- Shows actual and predicted labels.
- Identifies correct and incorrect predictions.
- Provides access to generated prompts and model responses.

### 🧠 Live Prompt Tester

- Allows users to test classification prompts interactively.
- Displays model predictions for user-provided text.

### ✍️ Custom Prompt Evaluation

- Supports user-defined classification instructions.
- Evaluates custom prompts with different prompting strategies.

### 📈 Strategy Comparison

- Compares Zero-Shot, Good Few-Shot and Poor Few-Shot.
- Helps analyze the influence of example relevance.

---

## 📂 Project Structure

```text
MTech_AIML_07/
|
|-- app.py
|-- style.css
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- .streamlit/
|   |-- config.toml
|
|-- src/
|   |-- __init__.py
|   |-- datasets_loader.py
|   |-- evaluator.py
|   |-- model_client.py
|   |-- prompt_engine.py
|
|-- results/
|   |-- latest_summary_.csv
|   |-- latest_details.csv
```

---

## 🚀 Installation and Setup

### Prerequisites

Make sure the following are installed:

- Python 3.10 or later
- Git
- Ollama
- VS Code or another code editor

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MTech_AIML_07.git
cd MTech_AIML_07
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Configure Ollama

Download Ollama:

https://ollama.com/

Pull the model:

```bash
ollama pull qwen2.5:3b
```

Verify the model:

```bash
ollama list
```

Make sure Ollama is running.

### Step 5: Launch Streamlit

From the project root, run:

```bash
streamlit run app.py
```

If your Streamlit entry point is located inside `src/`, use:

```bash
python -m streamlit run src/app.py
```

The application will open in your browser.

---

## 📖 How to Use

1. Launch the Streamlit application.
2. Select IMDb or AG News.
3. Set the evaluation sample size.
4. Configure examples per class.
5. Enter a random seed.
6. Run the evaluation.
7. Compare the three prompting strategies.
8. Inspect individual predictions.
9. Export and analyze the CSV results.

---

## ⚠️ Limitations

The current implementation has several limitations:

1. **Limited evaluation size:** The reported experiment uses 100 samples per dataset.
2. **Single LLM:** Only Qwen2.5:3b is evaluated.
3. **Lexical similarity:** TF-IDF does not fully capture semantic meaning.
4. **Hardware constraints:** Local inference can be time-consuming.
5. **Prompt sensitivity:** Model predictions may change with prompt wording or selected examples.
6. **Limited statistical validation:** The reported results use one random seed rather than repeated experimental trials.

---

## 🔮 Future Improvements

Potential improvements include:

- Evaluating larger test samples.
- Running experiments with multiple random seeds.
- Comparing additional local LLMs.
- Using sentence embeddings for semantic example selection.
- Comparing different numbers of few-shot examples.
- Adding confusion matrices and per-class metrics.
- Performing statistical significance testing.
- Measuring inference time and resource usage.
- Expanding evaluation to additional classification datasets.

---

## 🎓 Academic Significance

This project demonstrates:

- Practical implementation of LLM prompting strategies.
- Text classification without task-specific model fine-tuning.
- Relevance-based few-shot example selection.
- Reproducible evaluation using fixed random seeds.
- Comparative analysis across binary and multi-class classification tasks.
- Detailed analysis of model predictions and failure cases.

---

## 👥 Team & Institution

**Project:** PromptEval — Evaluating LLM Prompting Strategies

**Program:** M.Tech — Computer Science & Engineering

**Institution:** Birla Institute of Technology, Mesra, Ranchi

**Team Members:**
- Add team member names here.

**Faculty Guide:**
- Add faculty name here.

---

## 📚 References

1. [IMDb Dataset — Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)
2. [AG News Dataset — Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news)
3. [Ollama Documentation](https://docs.ollama.com/)
4. [Qwen2.5 Model](https://ollama.com/library/qwen2.5)
5. [Streamlit Documentation](https://docs.streamlit.io/)
6. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

<div align="center">

### ⭐ PromptEval

**Exploring how demonstration relevance influences Large Language Model classification performance.**

Made with ❤️ using Python, Streamlit and Ollama.

</div>
