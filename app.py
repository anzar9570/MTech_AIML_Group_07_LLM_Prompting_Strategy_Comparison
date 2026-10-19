from pathlib import Path
import sys
import traceback

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure imports work when Streamlit is launched from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets_loader import load_classification_dataset
from src.prompt_engine import get_label_names, build_custom_prompt
from src.model_client import generate_response
from src.evaluator import run_evaluation, save_results, evaluate_custom_prompt


# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="PromptEval | LLM Prompting Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_css():
    css_path = PROJECT_ROOT / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()


def label_for(dataset_name, label_id):
    labels = get_label_names(dataset_name)
    return labels.get(int(label_id), "Unknown")


def read_csv_if_exists(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def render_metric_cards(row):
    cols = st.columns(4)
    fields = [
        ("Accuracy", "accuracy"),
        ("Macro-Precision", "macro_precision"),
        ("Macro-Recall", "macro_recall"),
        ("Macro-F1", "macro_f1"),
    ]
    for col, (label, key) in zip(cols, fields):
        value = row.get(key, None)
        display = f"{float(value):.1%}" if pd.notna(value) else "—"
        with col:
            metric_card(label, display)


def format_metrics_table(df):
    if df.empty:
        return df
    result = df.copy()
    for column in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if column in result.columns:
            result[column] = result[column].apply(
                lambda value: f"{float(value):.2%}" if pd.notna(value) else "—"
            )
    return result


def run_single_prediction(dataset_name, text, instructions):
    prompt = build_custom_prompt(
        dataset_name=dataset_name,
        text=text,
        custom_instructions=instructions,
    )
    raw_response, response_time = generate_response(prompt)
    predicted_label_id = None
    # Use the same parser as the evaluator so UI and metrics are consistent.
    from src.prompt_engine import parse_prediction
    predicted_label_id = parse_prediction(raw_response, dataset_name)
    predicted_label = (
        label_for(dataset_name, predicted_label_id)
        if predicted_label_id is not None
        else "Invalid response"
    )
    return prompt, raw_response, predicted_label, response_time


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🧪 PromptEval")
    st.caption("LLM Prompting Evaluation Lab")
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Live Prompt Tester",
            "Custom Prompt Evaluation",
            "Run Strategy Comparison",
            "Results & Predictions",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Local model**")
    st.code("qwen2.5:3b · Ollama", language=None)
    st.caption("No paid API required. Ollama must be running locally.")


# ----------------------------- Header -----------------------------
st.markdown(
    '<div class="hero"><div class="eyebrow">M.Tech · AIML Project</div>'
    '<h1>PromptEval</h1>'
    '<p>Evaluate how prompt design affects text-classification performance.</p></div>',
    unsafe_allow_html=True,
)


# ----------------------------- Dashboard -----------------------------
if page == "Dashboard":
    st.subheader("Evaluation dashboard")
    st.write("Review saved results or run a new experiment from the navigation menu.")

    summary_path = RESULTS_DIR / "latest_summary.csv"
    details_path = RESULTS_DIR / "latest_details.csv"
    summary_df = read_csv_if_exists(summary_path)
    details_df = read_csv_if_exists(details_path)

    if summary_df.empty:
        st.info(
            "No saved standard-strategy results were found yet. "
            "Open **Run Strategy Comparison** to run zero-shot, good few-shot, and poor few-shot."
        )
    else:
        dataset_options = summary_df["dataset"].dropna().unique().tolist() if "dataset" in summary_df else []
        selected_dataset = st.selectbox("Dataset", dataset_options, key="dashboard_dataset")
        shown = summary_df[summary_df["dataset"] == selected_dataset].copy()

        if not shown.empty:
            best_row = shown.sort_values("accuracy", ascending=False).iloc[0]
            st.markdown(f"**Best accuracy in {selected_dataset}:** {best_row['strategy'].replace('_', ' ').title()}")
            render_metric_cards(best_row)

        st.markdown("### Strategy comparison")
        needed = {"strategy", "accuracy", "macro_precision", "macro_recall", "macro_f1"}
        if needed.issubset(shown.columns):
            chart_df = shown.melt(
                id_vars=["strategy"],
                value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1"],
                var_name="Metric",
                value_name="Score",
            )
            chart_df["Metric"] = chart_df["Metric"].map({
                "accuracy": "Accuracy",
                "macro_precision": "Macro-Precision",
                "macro_recall": "Macro-Recall",
                "macro_f1": "Macro-F1",
            })
            fig = px.bar(
                chart_df,
                x="strategy",
                y="Score",
                color="Metric",
                barmode="group",
                range_y=[0, 1],
                template="plotly_white",
                labels={"strategy": "Prompting strategy", "Score": "Score"},
            )
            fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Saved summary")
        st.dataframe(format_metrics_table(shown), use_container_width=True, hide_index=True)

        if not details_df.empty and {"dataset", "strategy", "correct"}.issubset(details_df.columns):
            detail_subset = details_df[details_df["dataset"] == selected_dataset]
            if not detail_subset.empty:
                correct_count = int(detail_subset["correct"].astype(str).str.lower().isin(["true", "1"]).sum())
                total_count = len(detail_subset)
                st.caption(f"Saved individual predictions for {selected_dataset}: {correct_count} correct out of {total_count} strategy predictions.")


# ----------------------------- Live Prompt Tester -----------------------------
elif page == "Live Prompt Tester":
    st.subheader("Live Prompt Tester")
    st.write("Enter your own classification instructions and a single review or news article.")
    dataset_name = st.selectbox("Dataset / label set", ["IMDb", "AG News"], key="live_dataset")
    labels = list(get_label_names(dataset_name).values())
    st.caption("Allowed labels: " + " · ".join(labels))

    default_instructions = (
        "Read the text carefully and identify its main sentiment. "
        "Choose Positive if the review expresses an overall favorable opinion, "
        "or Negative if it expresses an overall unfavorable opinion."
        if dataset_name == "IMDb"
        else
        "Identify the main topic of the news article. Choose World for international or political events, "
        "Sports for sporting events, Business for companies or markets, and Sci/Tech for science or technology."
    )
    instructions = st.text_area(
        "Your prompt instructions",
        value=default_instructions,
        height=130,
        key=f"live_instructions_{dataset_name}",
        help="Edit these instructions to test how wording changes the model's prediction.",
    )
    sample_text = st.text_area(
        "Text to classify",
        value="The film was beautifully acted and the story kept me interested until the end."
        if dataset_name == "IMDb"
        else "Researchers announced a new breakthrough in battery technology that could improve electric vehicles.",
        height=160,
        key=f"live_text_{dataset_name}",
    )
    true_label = st.selectbox(
        "Optional: known correct label",
        ["Not provided"] + labels,
        help="Select a known label to see whether this single prediction is correct. One example does not produce a meaningful overall accuracy score.",
    )

    if st.button("Classify text", type="primary", use_container_width=False):
        if not instructions.strip():
            st.error("Please enter prompt instructions.")
        elif not sample_text.strip():
            st.error("Please enter text to classify.")
        else:
            try:
                with st.spinner("Waiting for Qwen via Ollama..."):
                    prompt, raw_response, predicted_label, elapsed = run_single_prediction(
                        dataset_name, sample_text, instructions
                    )
                st.session_state["last_live_result"] = {
                    "dataset": dataset_name,
                    "text": sample_text,
                    "instructions": instructions,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "predicted_label": predicted_label,
                    "elapsed": elapsed,
                    "true_label": true_label,
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Check that Ollama is running and the model qwen2.5:3b is available.")

    result = st.session_state.get("last_live_result")
    if result and result.get("dataset") == dataset_name:
        st.markdown("### Prediction")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.caption("Predicted category")
            st.markdown(f"## {result['predicted_label']}")
            st.caption(f"Response time: {result['elapsed']:.2f} seconds")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.caption("Raw model response")
            st.write(result["raw_response"] or "(empty response)")
            if result["true_label"] != "Not provided":
                st.write("**Known label:**", result["true_label"])
                st.write("**Correct:**", "Yes" if result["predicted_label"] == result["true_label"] else "No")
            st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("Inspect the exact prompt sent to the model"):
            st.code(result["prompt"], language="text")


# ----------------------------- Custom Prompt Evaluation -----------------------------
elif page == "Custom Prompt Evaluation":
    st.subheader("Custom Prompt Evaluation")
    st.write("Evaluate instructions written by you or your faculty across a labelled test sample.")
    st.info(
        "Your instructions can be evaluated with zero-shot, good few-shot, or poor few-shot prompting. "
        "Demonstrations come only from the training split and always keep their original labels. "
        "The test labels are used only for scoring. Use the same sample size and random seed when comparing strategies."
    )

    dataset_name = st.selectbox("Dataset", ["IMDb", "AG News"], key="custom_eval_dataset")
    strategy_display = st.selectbox(
        "Prompting strategy",
        ["Zero-shot", "Good few-shot", "Poor few-shot"],
        key="custom_eval_strategy",
        help="Good few-shot uses the most lexically similar examples per class; poor few-shot uses the least similar examples per class.",
    )
    strategy_map = {
        "Zero-shot": "zero_shot",
        "Good few-shot": "good_few_shot",
        "Poor few-shot": "poor_few_shot",
    }
    selected_strategy = strategy_map[strategy_display]
    labels = list(get_label_names(dataset_name).values())
    st.caption("Allowed labels: " + " · ".join(labels))
    if dataset_name == "IMDb":
        starter = (
            "Classify the sentiment of the movie review. Focus on the overall opinion, not isolated words. "
            "Return Positive for a favorable review and Negative for an unfavorable review."
        )
    else:
        starter = (
            "Classify the news article by its main topic. Use World for world news, Sports for sports, "
            "Business for business and financial news, and Sci/Tech for science and technology."
        )
    instructions = st.text_area(
        "Custom prompt instructions",
        value=starter,
        height=150,
        key=f"custom_eval_instructions_{dataset_name}",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_size = st.number_input("Number of test texts", min_value=1, max_value=500, value=20, step=10, key="custom_eval_sample_size")
    with col2:
        random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="custom_eval_seed")
    with col3:
        examples_per_class = st.number_input(
            "Examples per class", min_value=1, max_value=5, value=1, step=1,
            disabled=(selected_strategy == "zero_shot"), key="custom_eval_examples_per_class",
        )

    st.caption("Larger samples take longer because every text requires a model call. Keep sample size and seed fixed for a fair strategy comparison.")
    if st.button("Evaluate custom prompt", type="primary"):
        if not instructions.strip():
            st.error("Please enter prompt instructions.")
        else:
            try:
                with st.spinner(f"Evaluating {sample_size} texts with Qwen. This may take a while..."):
                    custom_summary, custom_details = evaluate_custom_prompt(
                        dataset_name=dataset_name,
                        custom_instructions=instructions,
                        strategy=selected_strategy,
                        sample_size=int(sample_size),
                        random_seed=int(random_seed),
                        examples_per_class=int(examples_per_class),
                    )
                st.session_state["custom_summary"] = custom_summary
                st.session_state["custom_details"] = custom_details
                st.session_state["custom_run_info"] = {
                    "dataset": dataset_name,
                    "instructions": instructions,
                    "strategy": selected_strategy,
                    "strategy_display": strategy_display,
                    "sample_size": int(sample_size),
                    "random_seed": int(random_seed),
                    "examples_per_class": int(examples_per_class),
                }
            except Exception as exc:
                st.error(f"Custom evaluation failed: {exc}")
                st.info("Check Ollama is running, the backend update is installed, and the dataset can be loaded.")

    custom_summary = st.session_state.get("custom_summary")
    custom_details = st.session_state.get("custom_details")
    run_info = st.session_state.get("custom_run_info", {})
    if custom_summary is not None and run_info.get("dataset") == dataset_name and run_info.get("strategy") == selected_strategy:
        st.markdown(f"### Evaluation metrics — {run_info.get('strategy_display', 'Custom prompt')}")
        render_metric_cards(custom_summary.iloc[0])
        st.dataframe(format_metrics_table(custom_summary), use_container_width=True, hide_index=True)

        if custom_details is not None and not custom_details.empty:
            correct_count = int(custom_details["correct"].sum())
            st.caption(f"Correct predictions: {correct_count} / {len(custom_details)}")
            st.markdown("### Individual predictions")
            display_cols = [
                "sample_number", "true_label", "predicted_label",
                "correct", "valid_prediction", "response_time_seconds"
            ]
            st.dataframe(custom_details[display_cols], use_container_width=True, hide_index=True)
            with st.expander("Inspect prompts and selected demonstrations for each prediction"):
                sample_numbers = custom_details["sample_number"].tolist()
                chosen_sample = st.selectbox("Sample number", sample_numbers, key="custom_prompt_inspect_sample")
                selected_row = custom_details.loc[custom_details["sample_number"] == chosen_sample].iloc[0]
                st.write("**True label:**", selected_row["true_label"])
                st.write("**Predicted label:**", selected_row["predicted_label"])
                st.write("**Demonstration examples:**")
                demos = selected_row.get("demonstration_examples", "")
                st.code(demos if isinstance(demos, str) and demos.strip() else "No demonstrations (zero-shot).", language="text")
                st.write("**Exact prompt sent to the model:**")
                st.code(selected_row.get("prompt", "Prompt not logged in this result."), language="text")
            st.download_button(
                "Download prediction details (CSV)",
                data=custom_details.to_csv(index=False).encode("utf-8"),
                file_name=f"custom_prompt_{dataset_name.lower().replace('/', '-')}_details.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download metrics (CSV)",
                data=custom_summary.to_csv(index=False).encode("utf-8"),
                file_name=f"custom_prompt_{dataset_name.lower().replace('/', '-')}_metrics.csv",
                mime="text/csv",
            )
            with st.expander("View the custom instructions and experiment settings"):
                st.write("**Strategy:**", run_info.get("strategy_display", ""))
                st.write("**Sample size:**", run_info.get("sample_size", ""))
                st.write("**Random seed:**", run_info.get("random_seed", ""))
                st.code(run_info.get("instructions", ""), language="text")


# ----------------------------- Standard Strategy Evaluation -----------------------------
elif page == "Run Strategy Comparison":
    st.subheader("Run Strategy Comparison")
    st.write("Run the original experiment using the same test texts for all three strategies.")
    st.info(
        "Good few-shot selects the highest TF-IDF lexical-similarity examples per class. "
        "Poor few-shot selects the lowest-similarity examples per class. Both retain correct training labels."
    )

    dataset_choice = st.multiselect(
        "Datasets",
        ["IMDb", "AG News"],
        default=["IMDb", "AG News"],
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_size = st.number_input("Test sample size per dataset", min_value=1, max_value=500, value=20, step=10, key="strategy_sample_size")
    with col2:
        examples_per_class = st.number_input("Examples per class", min_value=1, max_value=5, value=1, step=1)
    with col3:
        random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="strategy_seed")

    st.caption("Each strategy makes one model call per test text. Runtime depends on your computer and Ollama.")
    if st.button("Run all three strategies", type="primary"):
        if not dataset_choice:
            st.error("Select at least one dataset.")
        else:
            try:
                with st.spinner("Running evaluation. Please keep this page open..."):
                    summary_df, details_df = run_evaluation(
                        dataset_names=dataset_choice,
                        sample_size=int(sample_size),
                        random_seed=int(random_seed),
                        examples_per_class=int(examples_per_class),
                    )
                    save_results(summary_df, details_df)
                st.session_state["standard_summary"] = summary_df
                st.session_state["standard_details"] = details_df
                st.success("Evaluation finished and results saved.")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.info("Check Ollama is running and the selected dataset is available.")

    summary_df = st.session_state.get("standard_summary")
    details_df = st.session_state.get("standard_details")
    if summary_df is not None:
        st.markdown("### Results from this run")
        st.dataframe(format_metrics_table(summary_df), use_container_width=True, hide_index=True)
        chart_df = summary_df.melt(
            id_vars=["dataset", "strategy"],
            value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1"],
            var_name="Metric",
            value_name="Score",
        )
        chart_df["Metric"] = chart_df["Metric"].str.replace("_", " ").str.title()
        fig = px.bar(
            chart_df,
            x="strategy",
            y="Score",
            color="Metric",
            facet_col="dataset",
            barmode="group",
            range_y=[0, 1],
            template="plotly_white",
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="prompteval_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download prediction details CSV",
            data=details_df.to_csv(index=False).encode("utf-8"),
            file_name="prompteval_details.csv",
            mime="text/csv",
        )


# ----------------------------- Results & Predictions -----------------------------
elif page == "Results & Predictions":
    st.subheader("Results & Predictions")
    summary_df = read_csv_if_exists(RESULTS_DIR / "latest_summary.csv")
    details_df = read_csv_if_exists(RESULTS_DIR / "latest_details.csv")

    if summary_df.empty and details_df.empty:
        st.info("No saved results yet. Run a strategy comparison first.")
    else:
        if not summary_df.empty:
            st.markdown("### Saved metrics")
            dataset_filter = st.multiselect(
                "Filter datasets",
                summary_df["dataset"].dropna().unique().tolist() if "dataset" in summary_df else [],
                default=summary_df["dataset"].dropna().unique().tolist() if "dataset" in summary_df else [],
                key="saved_dataset_filter",
            )
            filtered_summary = summary_df[summary_df["dataset"].isin(dataset_filter)] if "dataset" in summary_df else summary_df
            st.dataframe(format_metrics_table(filtered_summary), use_container_width=True, hide_index=True)
            st.download_button(
                "Download saved summary CSV",
                data=filtered_summary.to_csv(index=False).encode("utf-8"),
                file_name="latest_summary.csv",
                mime="text/csv",
            )
        if not details_df.empty:
            st.markdown("### Individual predictions")
            if "dataset" in details_df:
                ds_options = details_df["dataset"].dropna().unique().tolist()
                ds_filter = st.selectbox("Dataset for prediction details", ds_options)
                view_df = details_df[details_df["dataset"] == ds_filter]
            else:
                view_df = details_df
            if "strategy" in view_df:
                strategy_options = view_df["strategy"].dropna().unique().tolist()
                strategy_filter = st.multiselect("Strategies", strategy_options, default=strategy_options)
                view_df = view_df[view_df["strategy"].isin(strategy_filter)]
            if "correct" in view_df:
                correctness_filter = st.selectbox("Correctness", ["All", "Correct only", "Incorrect only"])
                if correctness_filter == "Correct only":
                    view_df = view_df[view_df["correct"].astype(str).str.lower().isin(["true", "1"])]
                elif correctness_filter == "Incorrect only":
                    view_df = view_df[~view_df["correct"].astype(str).str.lower().isin(["true", "1"])]
            st.dataframe(view_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered prediction details CSV",
                data=view_df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_prediction_details.csv",
                mime="text/csv",
            )

st.markdown(
    '<div class="footer">PromptEval · Local inference with Ollama · IMDb & AG News</div>',
    unsafe_allow_html=True,
)
