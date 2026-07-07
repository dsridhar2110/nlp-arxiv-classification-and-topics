"""
Phase-1 improvement: MULTI-LABEL classification.
Extends the original binary (ComputationalLinguistics) task to predict all THREE
subfield labels at once (one-vs-rest), using the best Phase-1 recipe: TF-IDF on
the abstract + Logistic Regression / Linear SVM. Reports F1 per label + macro/micro.
"""
import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score

LABELS = ["ComputationalLinguistics", "MachineLearning", "HumanComputerInteraction"]
INPUT = "abstract"   # the best-performing input from Phase 1

def load(split):
    df = pd.read_csv(f"data/{split}_set.csv")
    df[INPUT] = df[INPUT].fillna("")
    return df

print("Loading data...", flush=True)
tr, te = load("train"), load("test")
print(f"train {len(tr):,} | test {len(te):,}", flush=True)

print("Fitting TF-IDF on abstracts...", flush=True)
t0 = time.time()
vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                      ngram_range=(1, 2), min_df=3, max_features=50000)
Xtr = vec.fit_transform(tr[INPUT]); Xte = vec.transform(te[INPUT])
print(f"  TF-IDF {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)

models = {
    "LogReg": lambda: LogisticRegression(max_iter=1000, C=1.0),
    "LinearSVM": lambda: LinearSVC(C=1.0),
}

rows = []
for mname, mk in models.items():
    per_label = {}
    for lab in LABELS:
        ytr, yte = tr[lab].values, te[lab].values
        clf = mk(); clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        per_label[lab] = dict(
            f1=f1_score(yte, pred), precision=precision_score(yte, pred),
            recall=recall_score(yte, pred), positive_rate=float(yte.mean()))
        print(f"  [{mname}] {lab:26s} F1={per_label[lab]['f1']:.3f} "
              f"(P={per_label[lab]['precision']:.3f} R={per_label[lab]['recall']:.3f})", flush=True)
    f1s = [per_label[l]["f1"] for l in LABELS]
    # micro-F1 across all labels
    allpred, ally = [], []
    for lab in LABELS:
        clf = mk(); clf.fit(Xtr, tr[lab].values)
        allpred.append(clf.predict(Xte)); ally.append(te[lab].values)
    micro = f1_score(np.concatenate(ally), np.concatenate(allpred), average="micro")
    rows.append(dict(model=mname, per_label=per_label,
                     macro_f1=float(np.mean(f1s)), micro_f1=float(micro)))
    print(f"  [{mname}] MACRO-F1={np.mean(f1s):.3f} | MICRO-F1={micro:.3f}\n", flush=True)

json.dump(rows, open("ml/multilabel_results.json", "w"), indent=2)
print("="*56)
print("MULTI-LABEL LEADERBOARD (test F1, abstract input)")
print("="*56)
print(f"{'Label':28s}{'LogReg':>9s}{'LinearSVM':>11s}")
for lab in LABELS:
    lr = rows[0]["per_label"][lab]["f1"]; sv = rows[1]["per_label"][lab]["f1"]
    pr = rows[0]["per_label"][lab]["positive_rate"]
    print(f"{lab:28s}{lr:9.3f}{sv:11.3f}   ({pr*100:.0f}% positive)")
print("-"*48)
print(f"{'MACRO-F1 (avg of labels)':28s}{rows[0]['macro_f1']:9.3f}{rows[1]['macro_f1']:11.3f}")
print(f"{'MICRO-F1 (pooled)':28s}{rows[0]['micro_f1']:9.3f}{rows[1]['micro_f1']:11.3f}")
print("\nSaved -> ml/multilabel_results.json")
