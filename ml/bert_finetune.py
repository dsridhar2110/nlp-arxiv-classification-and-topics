"""
Phase-2 experiment: fine-tune a transformer (DistilBERT) on the binary
Computational-Linguistics task, and compare it head-to-head with the classical
TF-IDF + Logistic Regression baseline trained on the SAME data sample.
Uses Apple-Silicon MPS if available. Reports F1 on the test sample.
"""
import argparse, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

LABEL, INPUT, MODEL = "ComputationalLinguistics", "abstract", "distilbert-base-uncased"

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=15000)
ap.add_argument("--n-test", type=int, default=6000)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--maxlen", type=int, default=128)
args = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}", flush=True)

def load(split, n):
    df = pd.read_csv(f"data/{split}_set.csv")[[INPUT, LABEL]].dropna()
    df = df.rename(columns={INPUT: "text", LABEL: "label"})
    if n and n < len(df):
        _, df = train_test_split(df, test_size=n, stratify=df["label"], random_state=42)
    return df.reset_index(drop=True)

tr, te = load("train", args.n_train), load("test", args.n_test)
print(f"train {len(tr):,} | test {len(te):,} | positives train={tr.label.mean():.2f}", flush=True)

# ---- classical baseline on the SAME sample (fair comparison) ----
vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=3, max_features=50000)
Xtr = vec.fit_transform(tr.text); Xte = vec.transform(te.text)
lr = LogisticRegression(max_iter=1000).fit(Xtr, tr.label)
p = lr.predict(Xte)
tfidf_f1 = f1_score(te.label, p)
print(f"[TF-IDF+LogReg, same {len(tr)} sample] F1={tfidf_f1:.3f}", flush=True)

# ---- fine-tune DistilBERT ----
tok = AutoTokenizer.from_pretrained(MODEL)
def enc(texts): return tok(list(texts), truncation=True, max_length=args.maxlen, padding=True)
enc_tr, enc_te = enc(tr.text), enc(te.text)

class DS(Dataset):
    def __init__(self, e, y): self.e, self.y = e, list(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.e.items()}
        d["labels"] = torch.tensor(int(self.y[i])); return d

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
targs = TrainingArguments(output_dir="ml/_bert_out", num_train_epochs=args.epochs,
    per_device_train_batch_size=32, per_device_eval_batch_size=64, learning_rate=2e-5,
    logging_steps=50, save_strategy="no", report_to="none", disable_tqdm=False)
trainer = Trainer(model=model, args=targs, train_dataset=DS(enc_tr, tr.label))
t0 = time.time(); trainer.train(); print(f"fine-tuned in {(time.time()-t0)/60:.1f} min", flush=True)

pred = trainer.predict(DS(enc_te, te.label))
yhat = pred.predictions.argmax(-1)
bert_f1 = f1_score(te.label, yhat)
bert_p, bert_r = precision_score(te.label, yhat), recall_score(te.label, yhat)

res = {"n_train": len(tr), "n_test": len(te), "device": dev,
       "tfidf_logreg_f1_same_sample": round(float(tfidf_f1), 4),
       "bert_f1": round(float(bert_f1), 4), "bert_precision": round(float(bert_p), 4),
       "bert_recall": round(float(bert_r), 4), "tfidf_full_data_reference_f1": 0.920}
json.dump(res, open("ml/bert_results.json", "w"), indent=2)
print("\n" + "=" * 52)
print("PHASE 2 — TRANSFORMER vs CLASSICAL (same sample)")
print("=" * 52)
print(f"  TF-IDF + LogReg (same {len(tr)} docs) : F1 = {tfidf_f1:.3f}")
print(f"  DistilBERT (fine-tuned)             : F1 = {bert_f1:.3f}  (P={bert_p:.3f} R={bert_r:.3f})")
print(f"  [reference] TF-IDF+SVM on FULL 217k : F1 = 0.920")
print(f"\n  Verdict: BERT { 'WINS by' if bert_f1>tfidf_f1 else 'trails by' } {abs(bert_f1-tfidf_f1):.3f} vs same-sample classical")
print("Saved -> ml/bert_results.json")
