"""
源码文本特征：与论文「count-based n-gram」「TF-IDF」设定对齐的工程近似。

- NC：CountVectorizer，字符 n-gram（适合 Solidity 等无分词边界场景）
- TF-IDF：TfidfVectorizer，字符 n-gram
"""
from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def make_count_vectorizer(max_features: int = 40000, random_state: int = 42) -> CountVectorizer:
    return CountVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        min_df=2,
        dtype=float,
    )


def make_tfidf_vectorizer(max_features: int = 40000) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        min_df=2,
        dtype=float,
    )
