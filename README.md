# MTech_AIML_Group_07_LLM_Prompting_Strategy_Comparison

## 📊 Experimental Results

The project compares Zero-Shot, Good Few-Shot, and Poor Few-Shot prompting on IMDb and AG News.

### 📁 Result Files

- 📈 [Evaluation Summary](results/latest_summary_.csv)
- 🔍 [Individual Prediction Details](results/latest_details.csv)

### 🎯 Key Findings

- IMDb: Good Few-Shot and Poor Few-Shot both achieved 95% accuracy.
- AG News: Good Few-Shot achieved 78% accuracy, compared with 69% for Poor Few-Shot.
- On IMDb, 96 out of 100 predictions matched between the two few-shot strategies, while 4 differed.

**Experimental configuration:** 100 samples per dataset, random seed 42, local Qwen2.5:3b model.
