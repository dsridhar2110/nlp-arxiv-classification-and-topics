"""
Phase-2 (topics): BERTopic (transformer embeddings) vs LDA on the EXISTING arXiv
abstracts. Compares topic coherence (same c_v metric used for LDA), the number of
topics found, sample topics, and how well the discovered topics align with the
known subfield labels (a bonus check most unsupervised projects can't do).
"""
import argparse, json, time, warnings, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch

STOP = set(("a an the of and or to in for on with is are was were be been this that these those we our "
            "it its as by from at into over under between about across their they he she you i not but can "
            "which such using use used based new results model models method methods approach also show "
            "propose paper present two both more most than then when where while each other same").split())
def toks(t):
    return [w for w in re.findall(r"[a-z]{3,}", str(t).lower()) if w not in STOP]


def run(n):
    print(f"Loading {n:,} abstracts...", flush=True)
    df = pd.read_csv("data/train_set.csv")[["abstract", "ComputationalLinguistics",
         "MachineLearning", "HumanComputerInteraction"]].dropna(subset=["abstract"])
    df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
    docs = df["abstract"].tolist()
    texts = [toks(d) for d in docs]

    # ---- BERTopic (transformer embeddings + clustering) ----
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from hdbscan import HDBSCAN
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Embedding {len(docs):,} docs with MiniLM on {device}...", flush=True)
    emb_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    t0 = time.time()
    embeddings = emb_model.encode(docs, batch_size=128, show_progress_bar=True)
    print(f"  embedded in {(time.time()-t0)/60:.1f} min", flush=True)

    from sklearn.feature_extraction.text import CountVectorizer
    hdb = HDBSCAN(min_cluster_size=max(30, n // 400), metric="euclidean",
                  core_dist_n_jobs=1, prediction_data=True)   # single-threaded → no spawn
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)  # clean topic labels
    tm = BERTopic(nr_topics="auto", hdbscan_model=hdb, vectorizer_model=vec,
                  calculate_probabilities=False, verbose=False)
    topics, _ = tm.fit_transform(docs, embeddings)
    topic_ids = [t for t in sorted(tm.get_topics().keys()) if t != -1]
    n_bert = len(topic_ids)
    print(f"BERTopic found {n_bert} topics (+ outliers)", flush=True)
    bert_words = [[w for w, _ in tm.get_topic(t)][:10] for t in topic_ids]

    # ---- coherence (gensim c_v) ----
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel, LdaModel
    dct = Dictionary(texts); dct.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dct.doc2bow(t) for t in texts]
    def coherence(topic_words):
        tw = [[w for w in t if w in dct.token2id] for t in topic_words]
        tw = [t for t in tw if len(t) >= 3]
        if not tw: return float("nan")
        return CoherenceModel(topics=tw, texts=texts, dictionary=dct, coherence="c_v").get_coherence()
    bert_coh = coherence(bert_words)
    print(f"BERTopic coherence c_v = {bert_coh:.3f}", flush=True)

    # ---- LDA baseline at a comparable K ----
    K = int(min(max(n_bert, 10), 25))
    print(f"Training LDA with K={K}...", flush=True)
    lda = LdaModel(corpus=corpus, id2word=dct, num_topics=K, passes=5, random_state=42)
    lda_words = [[w for w, _ in lda.show_topic(i, topn=10)] for i in range(K)]
    lda_coh = coherence(lda_words)
    print(f"LDA coherence c_v = {lda_coh:.3f}", flush=True)

    # ---- label alignment: do topics specialise by subfield? ----
    df["topic"] = topics
    specialised = checked = 0
    for t in topic_ids:
        sub = df[df.topic == t]
        if len(sub) < 30: continue
        checked += 1
        dom = max(sub.ComputationalLinguistics.mean(), sub.MachineLearning.mean(), sub.HumanComputerInteraction.mean())
        if dom >= 0.7: specialised += 1
    align_pct = round(100 * specialised / checked, 1) if checked else 0.0

    res = {"n": n, "bertopic_topics": n_bert, "bertopic_coherence": round(float(bert_coh), 4),
           "lda_K": K, "lda_coherence": round(float(lda_coh), 4),
           "topics_dominated_by_one_subfield_pct": align_pct}
    json.dump(res, open(f"ml/topics_results_{n}.json", "w"), indent=2)

    print("\n" + "=" * 58)
    print(f"BERTopic vs LDA  ·  {n:,} arXiv abstracts")
    print("=" * 58)
    print(f"  LDA      (K={K:>3})            coherence c_v = {lda_coh:.3f}")
    print(f"  BERTopic ({n_bert:>3} topics)   coherence c_v = {bert_coh:.3f}")
    print(f"  Winner on coherence: {'BERTopic' if bert_coh > lda_coh else 'LDA'}")
    print(f"  {align_pct}% of BERTopic topics are dominated (>=70%) by one known subfield")
    print("\n  Sample BERTopic topics (top words):")
    for t in topic_ids[:6]:
        print(f"    Topic {t}: {', '.join(w for w, _ in tm.get_topic(t)[:7])}")
    print(f"\nSaved -> ml/topics_results_{n}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    run(ap.parse_args().n)
