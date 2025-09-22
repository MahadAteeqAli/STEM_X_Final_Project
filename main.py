# main.py
import os
import time
import pickle
import globals

from typing import List, Optional
from datetime import timedelta, datetime

from fastapi import FastAPI, HTTPException, Query, Depends, status, Body, Path
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select

from database import engine, create_db_and_tables
from models import User, Course, UserProfile, UserInteraction, QuizQuestion, QuizResponse
from auth import (
    get_password_hash, verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES,
)
from cbf import build_tfidf, recommend as cbf_recommend_core, load_tfidf
from cf import prepare_cf_data, train_cf_models, recommend_cf, popularity_topk, SVD_MODEL_PATH,  DATAFRAME_PATH
from mlp import predict_course_ratings, explain_course, _build_item_feature_matrix




with open("shap_explainer.pkl", "rb") as f:
    shap_explainer = pickle.load(f)


app = FastAPI(title="STEM Recommender", version="1.1")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ---------- Startup: ensure artifacts & tables ----------

@app.on_event("startup")
def startup_event():
    with Session(engine) as session:
        courses = session.exec(select(Course)).all()
        globals.tfidf_data = build_tfidf(courses or None)

    prepare_cf_data()
    train_cf_models()
    globals.svd_model = pickle.load(open(SVD_MODEL_PATH, "rb"))
    globals.svd_trainset = getattr(globals.svd_model, "trainset", None)
    print("✅ Preloaded SVD, trainset, and TF-IDF")


# ---------- Schemas ----------
class RecommendationItem(BaseModel):
    course_id: str
    name: str

class RecommendationsResponse(BaseModel):
    query: str
    recommendations: List[RecommendationItem]

class HybridResponse(BaseModel):
    query: Optional[str] = None
    cbf: List[RecommendationItem]
    cf: List[RecommendationItem]
    hybrid: List[RecommendationItem]

class PredictionItem(BaseModel):
    course_id: str
    score: float

class PredictResponse(BaseModel):
    predictions: List[PredictionItem]

class ShapItem(BaseModel):
    feature_idx: int
    value: float

class ExplainResponse(BaseModel):
    course_id: str
    shap_values: List[ShapItem]

class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class OnboardingAnswer(BaseModel):
    interests: Optional[str] = None
    domains: Optional[str] = None
    goals: Optional[str] = None
    level: Optional[int] = 1

class FeedbackIn(BaseModel):
    course_id: str
    event: str  # view/click/like/dislike/rating/enroll/complete
    rating: Optional[float] = None

# ---------- Auth ----------
@app.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(data: SignupRequest):
    with Session(engine) as session:
        exists = session.exec(
            select(User).where((User.username == data.username) | (User.email == data.email))
        ).first()
        if exists:
            raise HTTPException(400, "Username or email already registered")
        user = User(
            username=data.username,
            email=data.email,
            hashed_password=get_password_hash(data.password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        # 🔹 create a blank UserProfile automatically 
        profile = UserProfile(
            user_id=user.id,
            interests="",
            domains="",
            goals="",
            level=1,
            profile_vector=None
        )
        session.add(profile)
        session.commit()

        return {"message": "User created", "user_id": user.id}

@app.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(access_token=token)

# ---------- Onboarding & profile ----------
@app.get("/onboarding/questions")
def onboarding_questions(token: str = Depends(oauth2_scheme)):
    return {
        "questions": [
            {"key": "interests", "prompt": "What topics interest you? (comma-separated)"},
            {"key": "domains",   "prompt": "STEM domains of interest? (e.g., AI/ML, Data, Robotics)"},
            {"key": "goals",     "prompt": "What are your career goals or target roles?"},
            {"key": "level",     "prompt": "Your skill level? (1=beginner,2=intermediate,3=advanced)"},
        ]
    }

@app.post("/onboarding/submit")
def onboarding_submit(data: OnboardingAnswer, token: str = Depends(oauth2_scheme)):
    with Session(engine) as session:
        user = session.exec(select(User).order_by(User.id)).first()
        if not user:
            raise HTTPException(400, "No users exist; please /signup first.")
        prof = session.exec(select(UserProfile).where(UserProfile.user_id == user.id)).first()
        now = datetime.utcnow()

        # Build a simple content profile vector from TF-IDF vocab (query = interests + domains + goals)
        query_text = " ".join(filter(None, [data.interests, data.domains, data.goals])).strip()
        profile_vec_bytes = None
        if query_text:
            vectorizer, _, _ = load_tfidf()
            vec = vectorizer.transform([query_text]).toarray().astype("float32")[0]
            profile_vec_bytes = pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)

        if prof:
            prof.interests = data.interests
            prof.domains = data.domains
            prof.goals = data.goals
            prof.level = data.level or 1
            prof.profile_vector = profile_vec_bytes
            prof.updated_at = now
        else:
            prof = UserProfile(
                user_id=user.id,
                interests=data.interests,
                domains=data.domains,
                goals=data.goals,
                level=data.level or 1,
                profile_vector=profile_vec_bytes,
                created_at=now,
                updated_at=now,
            )
            session.add(prof)
        session.commit()
    return {"message": "Onboarding saved."}

# ---------- Feedback ----------
@app.post("/feedback")
def record_feedback(payload: FeedbackIn, token: str = Depends(oauth2_scheme)):
    with Session(engine) as session:
        user = session.exec(select(User).order_by(User.id)).first()
        if not user:
            raise HTTPException(400, "No users exist; please /signup first.")
        it = UserInteraction(
            user_id=user.id,
            course_id=str(payload.course_id),
            event=payload.event,
            rating=payload.rating,
            created_at=datetime.utcnow(),
        )
        session.add(it)
        session.commit()
    return {"message": "Feedback recorded."}

# ---------- CBF ----------
from database import get_db
from models import Course

@app.get("/recommend", response_model=RecommendationsResponse)
def cbf_recommend(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=20),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    recs = cbf_recommend_core(q=q, k=k)

    results = []
    for course_id in recs:
        course = db.query(Course).filter(Course.external_id == str(course_id)).first()
        results.append({
            "course_id": str(course_id),
            "name": course.title if course else "Unknown",
            "why": {
                "text": "explanation",
                "features": [
                    "."
                ]
            }
        })

    return {"query": q, "recommendations": results}

# ---------- CF ----------
@app.get("/recommend/cf")
def cf_recommend(
    user_id: str = Query(...),
    k: int = Query(5, ge=1, le=20),
    token: str = Depends(oauth2_scheme),
):
    recs = recommend_cf(user_id=user_id, k=k)
    results = []
    with Session(engine) as session:
        for course_id in recs:
            course = session.exec(select(Course).where(Course.external_id == str(course_id))).first()
            results.append({
                "course_id": str(course_id),
                "name": course.title if course else "Unknown",
                "why": {
                    "text": "explanation",
                    "features": [
                        "."
                    ]
                }
            })

    return {"recommendations": results}


# ---------- Personalized (cold-start aware) ----------
@app.get("/recommend/personalized")
def recommend_personalized(
    k: int = Query(5, ge=1, le=20),
    token: str = Depends(oauth2_scheme),
):
    with Session(engine) as session:
        user = session.exec(select(User).order_by(User.id)).first()
        if not user:
            raise HTTPException(400, "No users exist; please /signup first.")
        prof = session.exec(select(UserProfile).where(UserProfile.user_id == user.id)).first()

    # If no CF history => use profile + CBF + popularity
    seeds = []
    if prof and prof.interests:
        seeds.append(prof.interests)
    if prof and prof.domains:
        seeds.append(prof.domains)
    if prof and prof.goals:
        seeds.append(prof.goals)
    query = " ".join(seeds) if seeds else "STEM"

    cbf_ids = cbf_recommend_core(q=query, k=max(10, k))
    pop_ids = popularity_topk(max(10, k))
    # simple blend
    seen = set()
    blended = []
    for lst in (cbf_ids, pop_ids):
        for cid in lst:
            if cid not in seen:
                blended.append(cid)
                seen.add(cid)
    results = []
    with Session(engine) as session:
        for course_id in blended[:k]:
            course = session.exec(select(Course).where(Course.external_id == str(course_id))).first()
            results.append({
                "course_id": str(course_id),
                "name": course.title if course else "Unknown",
                "why": {
                    "text": "explanation",
                    "features": [
                        "."
                    ]
                }
            })

    return {"recommendations": results}


# ---------- Hybrid with MLP scoring ----------
@app.get("/recommend/hybrid", response_model=HybridResponse)
def hybrid_recommend(
    
    user_id: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    k: int = Query(5, ge=1, le=20),
    token: str = Depends(oauth2_scheme),
):
    start_total = time.time()
    print("➡️ Hybrid endpoint called")

    # Candidate generation
    base_q = q or "STEM"
    print(f"Base query: {base_q}")

    # get a larger pool from each model
    t0 = time.time()
    cbf_ids = cbf_recommend_core(q=base_q, k=50)
    print(f"✅ CBF got {len(cbf_ids)} items in {time.time() - t0:.2f}s")

    cf_ids = []
    if user_id is not None:
        try:
            t1 = time.time()
            cf_ids = recommend_cf(user_id=user_id, k=50)
            print(f"✅ CF got {len(cf_ids)} items in {time.time() - t1:.2f}s")
        except Exception as e:
            print(f"⚠️ CF failed: {e}")
            cf_ids = []

    #  Ensure hybrid has a true mix (CBF + CF union)
    if not cf_ids:
        print(" No CF results, falling back to CBF only")
    cand = list(dict.fromkeys(cbf_ids + cf_ids))
    print(f"Total candidate pool: {len(cand)}")

    # Score with MLP 
    try:
        t2 = time.time()
        ranked = predict_course_ratings(cand)
        print(f"✅ MLP scored {len(ranked)} items in {time.time() - t2:.2f}s")
        hybrid_ids = [cid for cid, _ in ranked[:k]]
        top_for_explain = hybrid_ids[:1]   

    except Exception as e:
        print(f"Hybrid MLP fallback: {e}")
        hybrid_ids = cand[:k]
        explanations = {}

    # DB lookup
    t3 = time.time()
    with Session(engine) as session:
        cbf_items = [
            RecommendationItem(
                course_id=str(cid),
                name=(session.exec(select(Course).where(Course.external_id == str(cid))).first() or Course(title="Unknown")).title
            )
            for cid in cbf_ids[:k]
        ]
        cf_items = [
            RecommendationItem(
                course_id=str(cid),
                name=(session.exec(select(Course).where(Course.external_id == str(cid))).first() or Course(title="Unknown")).title
            )
            for cid in cf_ids[:k]
        ]
        hybrid_items = [
            RecommendationItem(
                course_id=str(cid),
                name=(session.exec(select(Course).where(Course.external_id == str(cid))).first() or Course(title="Unknown")).title
            )
            for cid in hybrid_ids
        ]

    print(f"✅ DB lookups done in {time.time() - t3:.2f}s")

    print(f"🏁 Hybrid endpoint finished in {time.time() - start_total:.2f}s")

    return {
        "query": base_q,
        "cbf": cbf_items,
        "cf": cf_items,
        "hybrid": [
            {
                "course_id": item.course_id,
                "name": item.name,
                "why": {
                    "text": f"SHAP values for {item.course_id}",   
                    "plot": explanations.get(item.course_id, [])   
                }
            }
            for item in hybrid_items
        ]
    }

@app.get("/explain/course/{course_id}")
def explain_course_api(course_id: str = Path(...), token: str = Depends(oauth2_scheme)):
    return explain_course(course_id)
