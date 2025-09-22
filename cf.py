# cf.py
import os
import pickle
from typing import List, Dict, Tuple

import pandas as pd

# Attempt scikit-surprise; fallback to simple popularity & sklearn SVD-lite
_SURPRISE_OK = True
try:
    from surprise import Dataset, Reader, KNNBasic, SVD
    from surprise.model_selection import train_test_split
except Exception:
    _SURPRISE_OK = False
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    print("⚠️ Surprise not available; using sklearn fallback CF.")

# Paths
KNN_MODEL_PATH = "cf_knn.pkl"
SVD_MODEL_PATH = "cf_svd.pkl"
DATAFRAME_PATH = "cf_df.pkl"

# Filenames
REVIEWS_CSV = "Coursera_reviews.csv"
COURSES_CSV = "Coursera_courses.csv"

def prepare_cf_data(
    reviews_csv: str = REVIEWS_CSV,
    courses_csv: str = COURSES_CSV,
) -> None:
    """Load/prepare ratings; persist as ['user_id','course_id','rating']."""
    if not os.path.exists(courses_csv):
        raise FileNotFoundError(f"{courses_csv} not found.")
    courses = pd.read_csv(courses_csv, usecols=["course_id", "name"])
    courses["course_id"] = courses["course_id"].astype(str)

    if os.path.exists(reviews_csv):
        df = pd.read_csv(reviews_csv, usecols=["reviewers", "course_id", "rating"])
        df["reviewers"] = df["reviewers"].fillna("unknown")
        codes, _ = pd.factorize(df["reviewers"])
        df["user_id"] = (codes + 1).astype(str)

        df["course_id"] = df["course_id"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["course_id", "rating"])
        df["rating"] = df["rating"].astype(float)
        df = df[df["course_id"].isin(set(courses["course_id"]))]
        df = df[["user_id", "course_id", "rating"]]
        print(f"✅ Loaded {len(df)} valid ratings from {reviews_csv}")
    else:
        # neutral bootstrap ratings (keeps SVD happy)
        df = pd.DataFrame([("1", cid, 3.0) for cid in courses["course_id"]],
                          columns=["user_id", "course_id", "rating"])
        print(f"⚠️  {reviews_csv} not found; simulated {len(df)} neutral ratings")

    df.to_pickle(DATAFRAME_PATH)
    print(f"📦 Ratings saved to {DATAFRAME_PATH}")

def _train_surprise(df: pd.DataFrame):
    reader = Reader(rating_scale=(max(0.5, df.rating.min()), max(5.0, df.rating.max())))
    data = Dataset.load_from_df(df[["user_id", "course_id", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

    knn = KNNBasic(sim_options={"name": "cosine", "user_based": False})
    knn.fit(trainset)
    with open(KNN_MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)

    svd = SVD(n_factors=50, random_state=42)
    svd.fit(trainset)
    with open(SVD_MODEL_PATH, "wb") as f:
        pickle.dump(svd, f)

def _train_sklearn(df: pd.DataFrame):
    # very small implicit-MF-ish: pivot & SVD
    user_le, item_le = LabelEncoder(), LabelEncoder()
    u = user_le.fit_transform(df["user_id"].astype(str))
    i = item_le.fit_transform(df["course_id"].astype(str))
    r = df["rating"].values.astype(float)

    # build sparse-like dense (ok for small demo sets)
    n_u, n_i = u.max() + 1, i.max() + 1
    mat = np.zeros((n_u, n_i), dtype=float)
    for uu, ii, rr in zip(u, i, r):
        mat[uu, ii] = rr

    svd = TruncatedSVD(n_components=min(50, n_i-1), random_state=42)
    item_emb = svd.fit_transform(mat.T)  # items x k

    model = {
        "user_le": user_le,
        "item_le": item_le,
        "item_emb": item_emb,  # aligns to item_le.classes_
        "item_ids": item_le.classes_.tolist(),
        "item_pop": df.groupby("course_id")["rating"].count().sort_values(ascending=False),
    }
    with open(SVD_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def train_cf_models() -> None:
    df = pd.read_pickle(DATAFRAME_PATH)
    if _SURPRISE_OK:
        _train_surprise(df)
    else:
        _train_sklearn(df)
    print("✅ CF models saved.")

def popularity_topk(k: int = 5) -> List[str]:
    df = pd.read_pickle(DATAFRAME_PATH)
    pop = df.groupby("course_id")["rating"].count().sort_values(ascending=False)
    return pop.index.astype(str).tolist()[:k]

def recommend_cf(user_id: str, k: int = 5) -> List[str]:
    if _SURPRISE_OK and os.path.exists(SVD_MODEL_PATH):
        svd = pickle.load(open(SVD_MODEL_PATH, "rb"))
        # If Surprise, svd.trainset maps raw IDs
        try:
            inner_uid = svd.trainset.to_inner_uid(str(user_id))
        except Exception:
            # cold-start user => popularity
            return popularity_topk(k)
        # score all items not yet seen
        all_iids = svd.trainset.all_items()
        preds = []
        for inner_iid in all_iids:
            raw_iid = svd.trainset.to_raw_iid(inner_iid)
            est = svd.predict(str(user_id), raw_iid).est
            preds.append((raw_iid, est))
        preds.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in preds[:k]]
    else:
        # sklearn fallback: recommend by item popularity
        try:
            model = pickle.load(open(SVD_MODEL_PATH, "rb"))
            top = list(model["item_pop"].index.astype(str))[:k]
            return top
        except Exception:
            return popularity_topk(k)
