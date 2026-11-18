# Dynamic Model Selection — Adaptive LLM Routing Framework

## Overview
This project implements an adaptive routing prototype that classifies an input query into **LOW**, **MEDIUM**, or **HIGH** complexity and selects a corresponding local Qwen2.5 instruction model to generate a response.

The application combines NLP/text features, numerical and complexity features, and encoded metadata with a trained Linear SVM classifier. It also displays inference and routing measurements.

## Repository contents

| File | Description |
|---|---|
| `app.py` | Streamlit application, feature preparation, complexity prediction, routing and dashboard |
| `llm_engine.py` | Loads the selected Qwen2.5 model and generates a response |
| `final_complexity_classifier.pkl` | Trained complexity classifier |
| `final_tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `final_complexity_scaler.pkl` | Fitted scaler for complexity features |
| `final_numeric_scaler.pkl` | Fitted scaler for numerical features |
| `final_metadata_encoder.pkl` | Fitted metadata encoder |
| `Capstone_NLP_Preprocessing.ipynb` | Supplied NLP preprocessing/training notebook |
| `Dynamic_Model_Selection_Query_Metrics.xlsx` | Query-level metrics and experiment summary |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Common generated files excluded from Git |

## Model routing

| Predicted tier | Model |
|---|---|
| LOW | `Qwen/Qwen2.5-0.5B-Instruct` |
| MEDIUM | `Qwen/Qwen2.5-1.5B-Instruct` |
| HIGH | `Qwen/Qwen2.5-3B-Instruct` |

The models are downloaded from Hugging Face when first required. They are not bundled in this repository.

## Reported evaluation results

- Test accuracy: **67.33%**
- Macro F1-score: **65.97%**
- Normalized routing-cost saving on the 3,000-query evaluation: **40.43%**
- LLM generation-time reduction in a separate controlled three-query experiment: **26.76%**

**How to interpret these results:** Cost values are normalized relative units (1x / 2x / 4x), not currency or measured API charges. The 26.76% result refers to total LLM generation time in the controlled experiment, not end-to-end latency. End-to-end time can be higher because local model loading/switching adds overhead. The supplied metrics workbook contains the experiment details.

## Setup

Use Python 3.10 or 3.11 in a virtual environment where possible.

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU inference, install a PyTorch build compatible with your CUDA version using the official PyTorch installation selector. CPU inference may be substantially slower.

## Run the application

From the repository directory:

```bash
streamlit run app.py
```

The first use of each model may take longer because Hugging Face model files need to be downloaded. Ensure internet access is available and sufficient disk space and RAM/VRAM are available.

## Training notebook and data

`Capstone_NLP_Preprocessing.ipynb` is included as supplied. The raw training/test JSON data referenced by notebook cells are **not included** in this repository because they were not part of the supplied backup ZIP. To rerun the notebook end-to-end, provide those data files in the locations expected by the notebook.

The `.pkl` artifacts are included so the Streamlit application can load the trained classifier and preprocessing objects without retraining.

## Notes

- The application paths were made relative to the repository folder so the model artifacts can be found after cloning.
- The model artifacts are Python pickle/joblib files. Load only files from a trusted source.
- Actual API expenditure in the reported local-model experiment was ₹0 because inference was performed locally; this does not mean local inference has zero compute cost.
