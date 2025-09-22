# mlp.py
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import globals

from sklearn.model_selection import train_test_split
from cbf import load_tfidf
from cf import prepare_cf_data, DATAFRAME_PATH, SVD_MODEL_PATH

import globals

GLOBAL_SVD = globals.svd_model
GLOBAL_TRAINSET = globals.svd_trainset
GLOBAL_TFIDF = globals.tfidf_data

try:
    import shap
    _SHAP_OK = True
except Exception:
    _SHAP_OK = False
    shap = None
    print("⚠️ SHAP not available; explanations disabled.")


# Artifacts
MLP_MODEL_DIR = "mlp_model.keras"
SHAP_PKL      = "shap_explainer.pkl"

def prepare_mlp_features():
    vectorizer, tfidf_mat, tf_course_ids = load_tfidf()
    prepare_cf_data()
    df = pd.read_pickle(DATAFRAME_PATH)
    rated_ids = set(df["course_id"].astype(str).unique())

    svd = pickle.load(open(SVD_MODEL_PATH, "rb"))
    # Surprise SVD has .trainset & .qi; sklearn fallback stores dict without .qi
    trainset = getattr(svd, "trainset", None)
    qi = getattr(svd, "qi", None)

    common_ids, X_list = [], []
    for cid in tf_course_ids:
        if cid not in rated_ids:
            continue
        # TF-IDF
        idx = tf_course_ids.index(cid)
        tf_vec = tfidf_mat[idx].toarray().flatten()

        # CF embedding
        if trainset is not None and qi is not None:
            try:
                inner_iid = trainset.to_inner_iid(cid)
                cf_vec = svd.qi[inner_iid]
            except Exception:
                continue
        else:
            # sklearn fallback: no per-item latent; use zero vector of length 50
            cf_vec = np.zeros(50, dtype=float)

        X_list.append(np.concatenate([tf_vec, cf_vec]))
        common_ids.append(cid)

    X = np.vstack(X_list)
    return X, common_ids

def build_labels(course_ids):
    df = pd.read_pickle(DATAFRAME_PATH)
    df["course_id"] = df["course_id"].astype(str)
    mean_ratings = df.groupby("course_id")["rating"].mean()
    y = np.array([mean_ratings.get(cid, 0.0) for cid in course_ids], dtype=np.float32)
    return y

def train_mlp(epochs: int = 10, batch_size: int = 32):
    X, course_ids = prepare_mlp_features()
    y = build_labels(course_ids)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    model.save(MLP_MODEL_DIR)
    print(f"✅ MLP model saved to {MLP_MODEL_DIR}")

    if _SHAP_OK:
        explainer = shap.Explainer(model, X_train[:200])  
        with open(SHAP_PKL, "wb") as f:
            pickle.dump(explainer, f)
        print("✅ SHAP explainer saved.")

def _load_mlp():
    if not os.path.exists(MLP_MODEL_DIR):
        raise FileNotFoundError("MLP model not found. Train it via train_mlp().")
    model = tf.keras.models.load_model(MLP_MODEL_DIR)
    return model

def _build_item_feature_matrix(target_ids: list) -> np.ndarray:
    """Build TF-IDF + CF features for a list of courses (fast, no per-loop reloads)."""
    vectorizer, tfidf_mat, tf_course_ids = load_tfidf()
    rows = []

    for cid in target_ids:
        # --- TF-IDF vector ---
        try:
            idx = tf_course_ids.index(cid)
            tf_vec = tfidf_mat[idx].toarray().flatten()
        except ValueError:
            tf_vec = np.zeros(tfidf_mat.shape[1], dtype=float)

        # --- CF vector ---
        if GLOBAL_SVD and GLOBAL_TRAINSET:
            try:
                inner_iid = GLOBAL_TRAINSET.to_inner_iid(cid)
                cf_vec = GLOBAL_SVD.qi[inner_iid]
            except Exception:
                cf_vec = np.zeros(50, dtype=float)
        else:
            cf_vec = np.zeros(50, dtype=float)

        rows.append(np.concatenate([tf_vec, cf_vec]))

    return np.vstack(rows)

def predict_course_ratings(candidate_ids: list) -> list:
    """Return [(course_id, score)] predicted by the MLP, desc."""
    import time
    print("➡️ predict_course_ratings called with", len(candidate_ids), "candidates")
    t0 = time.time()
    model = _load_mlp()
    print(f"✅ Loaded MLP model in {time.time() - t0:.2f}s")

    t1 = time.time()
    X = _build_item_feature_matrix(candidate_ids)
    print(f"✅ Built feature matrix {X.shape} in {time.time() - t1:.2f}s")

    t2 = time.time()
    preds = model.predict(X, verbose=0).flatten()
    print(f"✅ MLP predicted {len(preds)} scores in {time.time() - t2:.2f}s")

    ranked = sorted(zip(candidate_ids, preds.tolist()), key=lambda x: x[1], reverse=True)
    print("➡️ predict_course_ratings finished in", time.time() - t0, "seconds")

    return ranked

import io, base64
import shap
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import io, base64

def _make_placeholder_plot() -> str:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, "No SHAP available", fontsize=12,
            ha="center", va="center")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def explain_course(course_id: str) -> dict:
    """Return SHAP explanation with summary text and bar plot for a course."""
    if not _SHAP_OK or not os.path.exists(SHAP_PKL):
            return {
                "course_id": course_id,
                "summary": "Default explanation: This course was chosen based on content similarity and popularity.",
                "top_features": [],
                "plot": _make_placeholder_plot()
            }

    with open(SHAP_PKL, "rb") as f:
        explainer = pickle.load(f)

    X = _build_item_feature_matrix([course_id])
    shap_values = explainer(X).values[0]

    # Get feature names (TF-IDF + CF)
    vectorizer, _, _ = load_tfidf()
    tfidf_features = vectorizer.get_feature_names_out()
    feature_names = list(tfidf_features) + [f"CF_dim{i}" for i in range(50)]

    # Top 5 influential features
    idxs = np.argsort(np.abs(shap_values))[::-1][:5]
    top_feats = [(feature_names[i], float(shap_values[i])) for i in idxs]

    # Textual explanation
    summary_lines = ["We recommended this course because:"]
    for name, val in top_feats:
        influence = "positive" if val > 0 else "negative"
        strength = "strong" if abs(val) > 0.5 else "moderate" if abs(val) > 0.2 else "slight"
        summary_lines.append(f"- {name} ({strength} {influence} influence)")
    summary = "\n".join(summary_lines)

    # Graphical explanation (horizontal bar chart)
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [val for _, val in reversed(top_feats)]
    labels = [name for name, _ in reversed(top_feats)]
    ax.barh(labels, vals, color=["#2ecc71" if v > 0 else "#e74c3c" for v in vals])
    ax.set_title("SHAP Feature Contributions")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    plot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "course_id": course_id,
        "summary": summary,
        "top_features": top_feats,
        "plot": plot_b64
    }

