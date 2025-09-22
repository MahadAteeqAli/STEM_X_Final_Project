# retrain.py
from cbf import build_tfidf
from cf import prepare_cf_data, train_cf_models
from mlp import train_mlp

def retrain_all():
    print("🔄 Rebuilding TF-IDF...")
    build_tfidf()

    print("🔄 Preparing CF data & training CF models...")
    prepare_cf_data()
    train_cf_models()

    print("🔄 Training MLP model & SHAP explainer...")
    train_mlp(epochs=10)

    return {"status": "ok"}

if __name__ == "__main__":
    print(retrain_all())
