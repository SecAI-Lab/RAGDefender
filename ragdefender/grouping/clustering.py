"""Clustering-based Grouping for single-hop QA — paper §4.1.

Hierarchical agglomerative clustering on passage embeddings, combined with a
TF-IDF top-``m`` term-frequency vote (Eq. 1) to decide whether the
adversarial passages form the smaller (``n_min``) or larger (``|R| − n_min``)
cluster (Eq. 2).
"""
from __future__ import annotations

import math
from typing import List, Optional

from ragdefender.grouping.base import Grouping


class ClusteringBasedGrouping(Grouping):
    """Single-hop QA grouping (paper §4.1). Datasets: NQ, MS MARCO."""

    def __init__(self, embedder, m: int = 5):
        """
        Args:
            embedder: A SentenceTransformer-like object exposing ``.encode(texts, convert_to_tensor=True)``.
            m: Top-``m`` TF-IDF terms used by the term-frequency vote (paper default: 5).
        """
        self.embedder = embedder
        self.m = m

    def estimate_n_adv(self, R: List[str], embeddings=None) -> int:
        if not R:
            return 0
        if embeddings is None:
            embeddings = self.embedder.encode(R, convert_to_tensor=True)

        # Agglomerative clustering on embeddings (paper §4.1: hierarchical clustering)
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(embeddings.cpu().detach().numpy())
        labels = list(model.labels_)
        num_labels = sum(labels)
        n_tfidf = self._n_tfidf(R)

        # Eq. 2 from the paper:
        #   N_adv = n_min                  if N_TF-IDF <= |R|/2
        #         = |R| - n_min            otherwise
        # where n_min is the smaller of the two cluster sizes.
        result = (
            min(num_labels, len(R) - num_labels)
            if num_labels > 0 and n_tfidf <= int(len(R) / 2)
            else max(num_labels, len(R) - num_labels)
        )
        return int(result)

    def _n_tfidf(self, R: List[str]) -> int:
        """Eq. 1 from paper §4.1.

        Compute the top-``m`` TF-IDF terms across ``R`` (with English stopwords
        removed); return the number of passages that contain more than ``m/2``
        of those terms.
        """
        import sklearn.feature_extraction.text as _text
        import pandas as pd

        stop_words = list(_text.ENGLISH_STOP_WORDS)
        tfidf = _text.TfidfVectorizer(stop_words=stop_words)
        X = tfidf.fit_transform(R)
        names = tfidf.get_feature_names_out()
        df = pd.DataFrame(X.todense().tolist(), columns=names)
        scores = df.T.sum(axis=1).sort_values(ascending=False)
        top_m = scores[: self.m]
        # Per-term presence indicators across R
        indices = [[1 if w in s else 0 for s in R] for w in top_m.index]
        # Vote: passage flagged if it contains more than floor(m/2) of the top terms
        votes = [
            sum(idx[i] for idx in indices) > math.floor(len(indices) / 2)
            for i in range(len(R))
        ]
        return sum(votes)
