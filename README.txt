# PromptEval UI files

Copy `app.py` and `style.css` into the project root, alongside the `src` folder and `results` folder.

Before running:
1. Install requirements in your project's virtual environment (`streamlit`, `pandas`, `plotly`, `scikit-learn`, `datasets`, `requests`).
2. Make sure the updated backend files from `PromptEval_backend_update.zip` are in `src/`.
3. Start Ollama and ensure `qwen2.5:3b` is available.

From the project root, run:
```bash
streamlit run app.py
```

The app uses local Ollama inference; no hosted paid API is used. Custom prompt evaluation is zero-shot evaluation over labelled test examples. It does not add few-shot demonstrations to the user-supplied prompt yet.
