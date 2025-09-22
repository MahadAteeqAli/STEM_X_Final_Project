# cbf.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------- Paths for artifacts (relative to project root) ----------
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
VECTORIZER_PKL = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
MATRIX_NPZ     = os.path.join(ARTIFACT_DIR, "tfidf_matrix.npz")
COURSE_IDS_PKL = os.path.join(ARTIFACT_DIR, "tfidf_course_ids.pkl")


@dataclass
class CourseRow:
    external_id: str
    title: str
    description: Optional[str]
    tags: Optional[str]


def _ensure_dirs() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _join_text(*parts: Optional[str]) -> str:
    return " ".join([p.strip() for p in parts if p and p.strip()])


# ---------------------------- Build / Save / Load ---------------------------

def build_tfidf(courses: Optional[Iterable[CourseRow]] = None):
    """
    Build TF-IDF from DB courses if available,
    otherwise fall back to Coursera_courses.csv.
    """
    import pandas as pd

    # Case 1: DB is empty
    if not courses:
        if not os.path.exists("Coursera_courses.csv"):
            print("⚠️ No courses in DB and no Coursera_courses.csv found → skipping TF-IDF")
            return None
        df = pd.read_csv("Coursera_courses.csv", usecols=["course_id", "name", "institution"])
        df["course_id"] = df["course_id"].astype(str)
        corpus = (df["name"].fillna("") + " " + df["institution"].fillna("")).tolist()
        course_ids = df["course_id"].tolist()
    else:
        # Case 2: Use DB courses
        course_rows: List[CourseRow] = list(courses)
        corpus = [_join_text(c.title, c.description, c.tags) for c in course_rows]
        course_ids = [c.external_id for c in course_rows]

    if not corpus:
        print("⚠️ Empty corpus after all sources → skipping TF-IDF")
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        lowercase=True
    )
    matrix = vectorizer.fit_transform(corpus)

    # Save artifacts for later use
    save_tfidf(vectorizer, matrix, course_ids)

    return vectorizer, matrix, course_ids

def save_tfidf(vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix, course_ids: List[int]) -> None:
    """
    Persist the TF-IDF artifacts to disk.
    """
    _ensure_dirs()
    # vectorizer
    with open(VECTORIZER_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    # sparse matrix
    sparse.save_npz(MATRIX_NPZ, matrix)
    # course id order
    with open(COURSE_IDS_PKL, "wb") as f:
        pickle.dump(course_ids, f)


def load_tfidf() -> Tuple[TfidfVectorizer, sparse.csr_matrix, List[int]]:
    """
    Load persisted TF-IDF artifacts. Raises FileNotFoundError if any part is missing.
    """
    if not (os.path.exists(VECTORIZER_PKL) and os.path.exists(MATRIX_NPZ) and os.path.exists(COURSE_IDS_PKL)):
        raise FileNotFoundError("TF-IDF artifacts not found. Run retraining to generate them.")

    with open(VECTORIZER_PKL, "rb") as f:
        vectorizer: TfidfVectorizer = pickle.load(f)
    matrix: sparse.csr_matrix = sparse.load_npz(MATRIX_NPZ)
    with open(COURSE_IDS_PKL, "rb") as f:
        course_ids: List[int] = pickle.load(f)

    # Safety: ensure CSR for fast row slicing
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()

    return vectorizer, matrix, course_ids


# ------------------------------ Recommender --------------------------------
def recommend(q: str, k: int = 10) -> List[int]:
    """
    Query => top-k course_ids using cosine similarity in TF-IDF space.
    """
    vectorizer, matrix, course_ids = load_tfidf()
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, matrix).ravel()
    if k <= 0:
        k = 10
    top_idx = np.argsort(-sims)[:k]
    return [course_ids[i] for i in top_idx]


