# app.py
import os
import httpx
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 15.0 

st.set_page_config(page_title="STEM Recommender", layout="wide")
title_map = {}  

def authed_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.get("token") else {}

def login_flow():
    st.header("Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Log In"):
        data = {"username": username, "password": password}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                r = client.post(f"{API_URL}/login", data=data, headers=headers)
            if r.status_code == 200:
                st.session_state.token = r.json().get("access_token")
                st.session_state.logged_in = True
                st.success("🔒 Logged in successfully!")
            else:
                st.error(f"❌ Login error: {r.text}")
        except Exception as e:
            st.error(f"❌ Request failed: {e}")

def signup_flow():
    st.header("Sign Up")
    username = st.text_input("Username", key="signup_user")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pass")

    if st.button("Create Account"):
        data = {"username": username, "email": email, "password": password}
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                r = client.post(f"{API_URL}/signup", json=data)
            if r.status_code == 201:
                st.success("✅ Account created! You can now log in.")
            else:
                st.error(f"❌ Signup error: {r.text}")
        except Exception as e:
            st.error(f"❌ Request failed: {e}")


def onboarding_ui():
    st.header("Onboarding")
    if not st.session_state.get("token"):
        st.warning("Please log in first.")
        return

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            q = client.get(f"{API_URL}/onboarding/questions", headers=authed_headers())
        st.caption("Answer a few quick questions to tailor your recommendations.")
        interests = st.text_input("Interests (comma-separated)")
        domains = st.text_input("STEM domains (e.g., AI/ML, Data)")
        goals = st.text_input("Career goals / target roles")
        level = st.selectbox("Skill level", [1, 2, 3], index=0)
        if st.button("Save Onboarding"):
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                r = client.post(f"{API_URL}/onboarding/submit",
                                json={"interests": interests, "domains": domains, "goals": goals, "level": level},
                                headers=authed_headers())
            st.success("✅ Onboarding saved." if r.status_code == 200 else f"❌ {r.text}")
    except Exception as e:
        st.error(f"❌ {e}")


def display_recs(recs: list, title: str = "Recommendations"):
    st.subheader(title)
    if not recs:
        st.write("No recommendations.")
        return

    for rec in recs:
        cid, name = rec.get("course_id"), rec.get("name", "Unknown")
        st.markdown(f"**{name}** (ID: {cid})")

        # Unique key for each button + state
        key = f"why_{title}_{cid}"
        if key not in st.session_state:
            st.session_state[key] = False

        if st.button(f"Why this? (ID: {cid})", key=f"btn_{title}_{cid}"):
            st.session_state[key] = not st.session_state[key]

        if st.session_state[key]:
            st.markdown("**Explanation:**")

            explanations = [
                "You completed Neural Networks and Deep Learning (strong positive influence)",
                "You enrolled in Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning (moderate positive influence)",
                "Learners who liked Crash Course on Python often moved into this course (moderate positive influence)",
                "Lack of completions in AI-heavy courses contributed negatively to your alignment here (slight negative influence)."
            ]

            for exp in explanations:
                st.markdown(f"- {exp}")


def call_cbf(q: str, k: int):
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(f"{API_URL}/recommend", params={"q": q, "k": k}, headers=authed_headers())
        if resp.status_code == 200:
            st.session_state["last_cbf_recs"] = resp.json().get("recommendations", [])
        else:
            st.error(f"❌ Error: {resp.text}")
    except Exception as e:
        st.error(f"❌ {e}")

    # Always render from session state
    if "last_cbf_recs" in st.session_state:
        display_recs(st.session_state["last_cbf_recs"], title="CBF Recommendations")

def call_cf(user_id: int, k: int):
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(f"{API_URL}/recommend/cf", params={"user_id": user_id, "k": k}, headers=authed_headers())
        if resp.status_code == 200:
            st.session_state["last_cf_recs"] = resp.json().get("recommendations", [])
        else:
            st.error(f"❌ Error: {resp.text}")
    except Exception as e:
        st.error(f"❌ {e}")

    if "last_cf_recs" in st.session_state:
        display_recs(st.session_state["last_cf_recs"], title="CF Recommendations")

def call_hybrid(user_id: int, q: str, k: int):
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(f"{API_URL}/recommend/hybrid",
                              params={"user_id": user_id, "q": q, "k": k},
                              headers=authed_headers())
        if resp.status_code == 200:
            data = resp.json()
            display_recs(data.get("cbf", []), title="CBF Recommendations")
            display_recs(data.get("cf", []), title="CF Recommendations")
            display_recs(data.get("hybrid", []), title="Hybrid (MLP-scored)")
        else:
            st.error(f"❌ Error: {resp.text}")
    except Exception as e:
        st.error(f"❌ {e}")

def feedback_ui():
    st.header("Feedback")
    if not st.session_state.get("token"):
        st.warning("Please log in first.")
        return
    course_id = st.text_input("Course ID")
    event = st.selectbox("Event", ["view", "click", "like", "dislike", "rating", "enroll", "complete"])
    rating = st.slider("Rating (if event=rating)", 1, 5, 5)
    if st.button("Submit Feedback"):
        body = {"course_id": course_id, "event": event, "rating": rating if event == "rating" else None}
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                r = client.post(f"{API_URL}/feedback", json=body, headers=authed_headers())
            if r.status_code == 200:
                st.success("✅ Feedback recorded.")
            else:
                st.error(f"❌ {r.text}")
        except Exception as e:
            st.error(f"❌ {e}")

def recommendation_ui():
    st.header("Get Recommendations")
    if not st.session_state.get("token"):
        st.warning("Please log in first.")
        return

    rec_type = st.selectbox("Recommendation Type", ["Content-Based", "Collaborative", "Hybrid", "Personalized"])
    k = st.number_input("Number of results (k)", min_value=1, max_value=20, value=5)

    if rec_type == "Content-Based":
        q = st.text_input("Enter your interests (e.g., 'Data Science')", key="cbf_q")
        if st.button("Get CBF Recommendations"):
            call_cbf(q, k)
    elif rec_type == "Collaborative":
        user_id = st.number_input("Your User ID", min_value=1, value=1, key="cf_user")
        if st.button("Get CF Recommendations"):
            call_cf(user_id, k)
    elif rec_type == "Hybrid":
        user_id = st.number_input("Your User ID", min_value=1, value=1, key="hy_user")
        q = st.text_input("Enter your interests", key="hy_q")

        if st.button("Get Hybrid Recommendations"):
            # 🔹 Save results into session_state so they persist after rerun
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(f"{API_URL}/recommend/hybrid",
                                params={"user_id": user_id, "q": q, "k": k},
                                headers=authed_headers())
            if resp.status_code == 200:
                st.session_state["last_hybrid_recs"] = resp.json()
            else:
                st.error(f"❌ Error: {resp.text}")

        # 🔹 Always render from session_state if available
        if "last_hybrid_recs" in st.session_state:
            data = st.session_state["last_hybrid_recs"]
            display_recs(data.get("cbf", []), title="CBF Recommendations")
            display_recs(data.get("cf", []), title="CF Recommendations")
            display_recs(data.get("hybrid", []), title="Hybrid (MLP-scored)")

    else:
        if st.button("Get Personalized"):
            try:
                with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                    r = client.get(f"{API_URL}/recommend/personalized", params={"k": k}, headers=authed_headers())
                if r.status_code == 200:
                    st.session_state["last_personalized_recs"] = r.json().get("recommendations", [])
                else:
                    st.error(f"❌ {r.text}")
            except Exception as e:
                st.error(f"❌ {e}")

        if "last_personalized_recs" in st.session_state:
            display_recs(st.session_state["last_personalized_recs"], title="Personalized (Cold-start aware)")

def main():
    st.title("STEM Recommender")
    if "token" not in st.session_state:
        st.session_state.token = None
        st.session_state.logged_in = False

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Login", "Signup", "Onboarding", "Recommend", "Feedback"])

    if page == "Login":
        login_flow()
    elif page == "Signup":
        signup_flow()
    elif page == "Onboarding":
        onboarding_ui()
    elif page == "Recommend":
        recommendation_ui()
    else:
        feedback_ui()

if __name__ == "__main__":
    main()
