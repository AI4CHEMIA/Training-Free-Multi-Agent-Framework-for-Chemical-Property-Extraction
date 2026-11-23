"""
data/index.py — FAISS retrieval index over training documents.

Uses OpenAI embeddings when available, otherwise TF-IDF.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config

INDEX_PATH = config.DATA_DIR / "faiss_index.bin"
META_PATH  = config.DATA_DIR / "faiss_meta.pkl"
TFIDF_PATH = config.DATA_DIR / "tfidf_vectorizer.pkl"

TFIDF_DIM = 512


class PropertyIndex:
    def __init__(self):
        self.index:      Optional[faiss.IndexFlatIP] = None
        self.metadata:   list[dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.dim:        int = TFIDF_DIM
        self._use_openai: bool = False

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, train_docs: list) -> None:
        texts = [d.text for d in train_docs]
        metas = [{
            "doi":        d.doi,
            "text":       d.text,
            "prop_ids":   d.prop_ids,
            "extractions": [e.to_dict() for e in d.extractions],
        } for d in train_docs]

        # Try OpenAI embeddings first
        vecs = self._openai_embed(texts)

        if vecs:
            self._use_openai = True
            self.dim = len(vecs[0])
            X = np.array(vecs, dtype="float32")
        else:
            self._use_openai = False
            self.dim = TFIDF_DIM
            self.vectorizer = TfidfVectorizer(
                max_features=TFIDF_DIM, ngram_range=(1, 2),
                sublinear_tf=True, strip_accents="unicode",
            )
            X = self.vectorizer.fit_transform(texts).toarray().astype("float32")

        X = normalize(X, norm="l2")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(X)
        self.metadata = metas
        print(f"[Index] Built: {self.index.ntotal} docs, dim={self.dim}, "
              f"backend={'openai' if self._use_openai else 'tfidf'}")

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from utils.llm_client import embed
            vecs = embed(texts)
            if vecs and len(vecs) == len(texts):
                return vecs
        except Exception as e:
            print(f"[Index] OpenAI embedding unavailable ({e}), using TF-IDF.")
        return []

    def _query_vec(self, text: str) -> np.ndarray:
        if self._use_openai:
            vecs = self._openai_embed([text])
            if vecs:
                v = np.array(vecs[0], dtype="float32").reshape(1, -1)
                return normalize(v, norm="l2")
        # TF-IDF fallback
        v = self.vectorizer.transform([text]).toarray().astype("float32")
        return normalize(v, norm="l2")

    # ── Query ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        assert self.index is not None, "Build the index first."
        q = self._query_vec(query)
        dists, idxs = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < 0:
                continue
            m = dict(self.metadata[idx])
            m["similarity"] = float(dist)
            results.append(m)
        return results

    def property_embedding(self, prop_id: int) -> np.ndarray:
        """Average embedding vector for a property (for Figure 7)."""
        vecs = []
        for meta in self.metadata:
            if prop_id in meta["prop_ids"]:
                v = self._query_vec(meta["text"])[0]
                vecs.append(v)
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.dim, dtype="float32")

    def property_distance(self, pid_a: int, pid_b: int) -> float:
        ea, eb = self.property_embedding(pid_a), self.property_embedding(pid_b)
        na, nb = np.linalg.norm(ea), np.linalg.norm(eb)
        if na == 0 or nb == 0:
            return 1.0
        return float(1 - np.dot(ea, eb) / (na * nb))

    # ── Persist ───────────────────────────────────────────────────────────

    def save(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH,  "wb") as f:
            pickle.dump((self.metadata, self.dim, self._use_openai), f)
        if self.vectorizer:
            with open(TFIDF_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)
        print(f"[Index] Saved → {INDEX_PATH}")

    @classmethod
    def load(cls) -> "PropertyIndex":
        idx = cls()
        idx.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            idx.metadata, idx.dim, idx._use_openai = pickle.load(f)
        if not idx._use_openai and TFIDF_PATH.exists():
            with open(TFIDF_PATH, "rb") as f:
                idx.vectorizer = pickle.load(f)
        print(f"[Index] Loaded: {idx.index.ntotal} docs")
        return idx

    @classmethod
    def load_or_build(cls, train_docs: list) -> "PropertyIndex":
        if INDEX_PATH.exists() and META_PATH.exists():
            try:
                return cls.load()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"[Index] Failed to load existing index ({e}), rebuilding..."
                )
        idx = cls()
        idx.build(train_docs)
        idx.save()
        return idx
