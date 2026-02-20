# arXiv CS Text Classification and Topic Modelling

**Binary classification** (Computational Linguistics vs. other) and **LDA topic modelling** on arXiv computer science articles (titles and abstracts, 2016–2025).

**Author:** Deekshita  
**Student ID:** 34051201  

---

## About the project

This repository contains:

1. **Part 1 — Text classification**  
   Predict whether an article is tagged **Computational Linguistics** using only the **Abstract** or only the **Title**.  
   - **Models:** Logistic Regression, SVM, RNN, LSTM (TF-IDF for LogReg and SVM).  
   - **Settings:** Two training sizes (first 1,000 samples vs full training set).  
   - **Outputs:** F1, precision, recall, accuracy, and precision–recall curves on the test set.

2. **Part 2 — Topic modelling**  
   Discover topics in the training corpus (title + abstract) with **LDA** (gensim).  
   - **Variants:** With vs without bigrams; number of topics K=10 and K=40.  
   - **Corpus sizes:** First 1,000 and first 20,000 documents.  
   - **Outputs:** Topic word lists, coherence scores, pyLDAvis, word clouds, t-SNE.

All experiments and figures are in the Jupyter notebook. A **Streamlit web app** gives an overview and lets you try the classifier (if a model is saved).

---

## Repository structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore
├── GITHUB_SETUP.md              # Steps to push to GitHub
│
├── code_34051201.ipynb          # Main notebook (submission filename)
├── arxiv_cs_classification_and_topic_modelling.ipynb   # Same notebook (descriptive name)
│
├── app/                         # Web UI
│   ├── streamlit_app.py         # Streamlit dashboard + Try classifier
│   └── results_summary.json     # Classification metrics (displayed in UI)
│
├── data/                        # Put train/dev/test CSVs here (not committed)
└── logs/                        # Classification logs (created when running notebook)
```

---

## Data

- **Source:** arXiv.org, computer science category (2016–2025).  
- **Files:** `train_set.csv`, `dev_set.csv`, `test_set.csv` (columns: Title, Abstract, HumanComputerInteraction, MachineLearning, ComputationalLinguistics).  
- **Usage:** Place the three CSVs in the project folder (or in `data/` and set `data_dir` in the notebook). These files are not included in the repo.

---

## Setup

**Requirements:** Python 3.8+

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/arxiv-cs-classification-lda.git
   cd arxiv-cs-classification-lda
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m nltk.download punkt wordnet stopwords punkt_tab
   ```

4. **Add data**  
   Put `train_set.csv`, `dev_set.csv`, and `test_set.csv` in the project root (or in `data/` and point the notebook to them).

---

## How to run

### Full pipeline (notebook)

Open and run all cells in one of the notebooks:

- `code_34051201.ipynb` or  
- `arxiv_cs_classification_and_topic_modelling.ipynb`

```bash
jupyter notebook code_34051201.ipynb
```

Part 1 trains multiple classifiers and may take a while on full data. Part 2 runs LDA for different K and corpus sizes.

### Web UI (Streamlit)

From the **project root**:

```bash
streamlit run app/streamlit_app.py
```

- **Home:** Project and data description, how to run.  
- **Part 1: Classification:** Setup and results table (from `app/results_summary.json`).  
- **Part 2: Topic Modelling:** LDA setup and pointer to the notebook.  
- **Try classifier:** Live prediction (requires saving a model from the notebook; see instructions in the app).

---

## Results summary

- **Classification:** Test-set F1, precision, recall, and accuracy for LogReg, SVM, RNN, and LSTM on Abstract and Title, with 1k and full training. Abstract input generally outperforms title (e.g. SVM F1 0.92 vs 0.87; LSTM 0.92 vs 0.91).  
- **Topic modelling:** Topic lists, coherence, pyLDAvis, word clouds, and t-SNE for 1k/20k docs and K=10/40 with and without bigrams.

Detailed analysis is in the report PDF (`report_34051201.pdf`), not in this repo.

---

## Tech stack

- **NLP:** spaCy, NLTK (tokenization, lemmatization, stopwords).  
- **Classification:** scikit-learn (TF-IDF, Logistic Regression, SVM), TensorFlow (tokenization/padding), PyTorch (RNN, LSTM).  
- **Topic modelling:** gensim (LDA), pyLDAvis.  
- **UI:** Streamlit.

See `requirements.txt` for versions.

---

## License

MIT (or as required by your course).
