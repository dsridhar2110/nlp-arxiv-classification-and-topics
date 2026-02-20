"""
Streamlit UI — arXiv CS Text Classification and Topic Modelling
Run from repo root: streamlit run app/streamlit_app.py
"""

import streamlit as st
import os
import json
import re

# Preprocessing must match the notebook (TF-IDF was fit on preprocessed text)
def _preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        return text.strip().lower()  # fallback if NLTK not available
    tech = re.findall(r"\b[A-Za-z]+(?:-\d+|\d+|-[A-Za-z]+)\b", text)
    tech = [t.lower().replace("-", "_") for t in tech]
    txt = text.lower()
    txt = re.sub(r"\b\d+\b", "", txt)
    txt = re.sub(r"[^\w\s-]", " ", txt).replace("-", "_")
    tokens = word_tokenize(txt)
    stop = set(stopwords.words("english"))
    tokens = [w for w in tokens if (w not in stop or w in tech)]
    lemma = WordNetLemmatizer()
    tokens = [w if "_" in w else lemma.lemmatize(w) for w in tokens]
    tokens.extend(tech)
    seen = set()
    return " ".join(x for x in tokens if not (x in seen or seen.add(x)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(ROOT, "app", "results_summary.json")

st.set_page_config(
    page_title="arXiv CS Classification & LDA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("📄 Project")
st.sidebar.markdown("**Deekshita** · 34051201")
st.sidebar.markdown("---")
st.sidebar.markdown("### Contents")
st.sidebar.markdown("- **Home** — Project description")
st.sidebar.markdown("- **Part 1** — Classification setup & results")
st.sidebar.markdown("- **Part 2** — Topic modelling setup")
st.sidebar.markdown("- **Try it** — Live classifier (if model saved)")
st.sidebar.markdown("---")
st.sidebar.markdown("### Run full pipeline")
st.sidebar.code("jupyter notebook\n# open code_34051201.ipynb", language="bash")
st.sidebar.markdown("---")
st.sidebar.caption("Notebook: `code_34051201.ipynb` or `arxiv_cs_classification_and_topic_modelling.ipynb`")

# Main title
st.title("arXiv CS Text Classification and Topic Modelling")
st.caption("Binary classification (Computational Linguistics) and LDA topic modelling on arXiv computer science abstracts (2016–2025)")

tab_home, tab_part1, tab_part2, tab_try = st.tabs(["🏠 Home", "📊 Part 1: Classification", "📚 Part 2: Topic Modelling", "🔮 Try classifier"])

# ---- Tab 1: Home ----
with tab_home:
    st.header("Project description")
    st.markdown("""
    This project has **two parts**, both using arXiv computer science articles (titles and abstracts):

    1. **Part 1 — Text classification**  
       Predict whether an article is tagged **Computational Linguistics** (binary: yes/no).  
       - **Input:** Abstract only, or Title only (separate experiments).  
       - **Models:** Logistic Regression, SVM, RNN, LSTM (TF-IDF for the first two).  
       - **Data size:** Trained on first 1,000 samples and on the full training set.  
       - **Outputs:** F1, precision, recall, accuracy, and precision–recall curves on the test set.

    2. **Part 2 — Topic modelling**  
       Discover topics in the training text (title + abstract) using **LDA** (gensim).  
       - **Variants:** With vs without bigrams; number of topics K=10 and K=40.  
       - **Data size:** First 1,000 and first 20,000 documents.  
       - **Outputs:** Topic words, coherence, pyLDAvis, word clouds, t-SNE.

    All code, experiments, and figures are in the **Jupyter notebook**.  
    This UI gives a high-level overview and, if you save a model, a way to try the classifier.
    """)

    st.header("Data")
    st.markdown("""
    - **Source:** arXiv.org, computer science category (2016–2025).  
    - **Fields:** Title, Abstract, and three binary labels (HumanComputerInteraction, MachineLearning, ComputationalLinguistics).  
    - **Splits:** train / dev / test CSVs (place in `data/` or current directory for the notebook).
    """)

    st.header("How to run the full project")
    st.markdown("""
    1. **Setup:** Install dependencies (`pip install -r requirements.txt`), download spaCy and NLTK data.  
    2. **Data:** Put `train_set.csv`, `dev_set.csv`, `test_set.csv` in the project folder (or `data/` and set paths in the notebook).  
    3. **Notebook:** Open `code_34051201.ipynb` (or `arxiv_cs_classification_and_topic_modelling.ipynb`) and run all cells.  
    4. **This UI:** Run `streamlit run app/streamlit_app.py` from the project root for this dashboard.
    """)

# ---- Tab 2: Part 1 ----
with tab_part1:
    st.header("Part 1: Text classification")
    st.markdown("Binary prediction of **ComputationalLinguistics** from Abstract or Title. Configurations: 2 inputs × 2 algorithms × 2 training sizes.")

    # Try to load results summary (optional file you can create from the notebook)
    if os.path.isfile(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                data = json.load(f)
            st.subheader("Results summary (from notebook)")
            if "table" in data:
                import pandas as pd
                st.dataframe(pd.DataFrame(data["table"]), use_container_width=True)
            if "note" in data:
                st.caption(data["note"])
        except Exception as e:
            st.caption(f"Could not load results file: {e}")
    else:
        st.info("""
        **To show a results table here:**  
        In the notebook, after evaluation, save a summary to `app/results_summary.json`.  
        Example format: `{"table": [{"Model": "LogReg", "Input": "Abstract", "Size": "full", "F1": 0.88, "Precision": 0.90, "Recall": 0.86}]}`
        """)
        st.markdown("**Configurations:**")
        st.markdown("""
        | Input   | Algorithms        | Training size |
        |---------|-------------------|----------------|
        | Abstract | LogReg, SVM, RNN, LSTM | 1,000 / full |
        | Title    | LogReg, SVM, RNN, LSTM | 1,000 / full |
        """)
    st.markdown("Full metrics and precision–recall curves are in the notebook.")

# ---- Tab 3: Part 2 ----
with tab_part2:
    st.header("Part 2: Topic modelling")
    st.markdown("LDA (gensim) on training text (title + abstract). Two main variations × two corpus sizes.")
    st.markdown("""
    | Variation   | Bigrams | K (topics) | Corpus size |
    |-------------|---------|------------|--------------|
    | 1           | Yes/No  | 10, 40     | 1,000 docs   |
    | 2           | Yes/No  | 10, 40     | 20,000 docs  |
    """)
    st.markdown("Outputs in the notebook: topic word lists, coherence scores, pyLDAvis, word clouds, t-SNE. Discussion is in the report PDF.")

# ---- Tab 4: Try classifier ----
with tab_try:
    st.header("Try the classifier")
    st.markdown("Enter a title or abstract to predict whether it would be tagged **Computational Linguistics**.")

    with st.expander("How it works / How to test"):
        st.markdown("""
        **How it works**
        1. You type a title or abstract in the box below.
        2. The app preprocesses it the same way as the notebook (lowercase, lemmatize, stopwords, etc.).
        3. It loads the saved TF-IDF vectorizer and Logistic Regression model from `models/` (trained on **abstracts**, full data).
        4. It shows the prediction (Computational Linguistics vs Other) and the probability.

        **How to test it**
        1. **Save a model from the notebook** (run once):  
           Add a **new cell** after your data and `tfidf_vectorize` are defined, then run:
           ```python
           import joblib
           field = "processed_abstract"
           X_tr, X_dev, X_te, y_tr, y_dev, y_te, vectorizer = tfidf_vectorize(field, df_train, df_dev, df_test, 10000)
           model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
           model.fit(X_tr, y_tr)
           os.makedirs("models", exist_ok=True)
           joblib.dump(vectorizer, "models/tfidf_abstract.joblib")
           joblib.dump(model, "models/logreg_abstract_full.joblib")
           print("Saved. You can now use the Try classifier tab.")
           ```
        2. **Run the app** from project root: `streamlit run app/streamlit_app.py`
        3. Open the **Try classifier** tab, paste a title or abstract, click **Predict**.
        """)

    input_text = st.text_area(
        "Title or abstract",
        placeholder="e.g. We propose a transformer-based model for neural machine translation with cross-lingual alignment...",
        height=120,
    )

    if st.button("Predict"):
        if not input_text.strip():
            st.warning("Enter some text.")
        else:
            models_dir = os.path.join(ROOT, "models")
            vectorizer_path = os.path.join(models_dir, "tfidf_abstract.joblib")
            model_path = os.path.join(models_dir, "logreg_abstract_full.joblib")

            if os.path.isfile(vectorizer_path) and os.path.isfile(model_path):
                try:
                    import joblib
                    vectorizer = joblib.load(vectorizer_path)
                    model = joblib.load(model_path)
                    preprocessed = _preprocess_text(input_text)
                    X = vectorizer.transform([preprocessed])
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
                    label = "Computational Linguistics" if pred == 1 else "Other"
                    st.success(f"**Prediction:** {label}")
                    if prob is not None:
                        st.metric("Probability (CompLing)", f"{prob:.2%}")
                except Exception as e:
                    st.error(f"Model loading or prediction failed: {e}")
            else:
                st.info(
                    "No saved model found. Save a TF-IDF vectorizer and classifier from the notebook to `models/` "
                    "as `tfidf_abstract.joblib` and `logreg_abstract_full.joblib` (see **How it works / How to test** above)."
                )
