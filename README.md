# STEM Course Recommender

An explainable hybrid recommender system for personalized STEM course suggestions, optimized for cold-start scenarios and final-year undergraduate student upskilling.

---

## Table of Contents

1. [Overview](#-overview)  
2. [Tech Stack](#-tech-stack)  
3. [Setup & Installation](#️-setup--installation)  
4. [Configuration](#-configuration) 
5. [Database Setup (PostgreSQL)](#-database-setup-postgresql)    
6. [Running the Application](#️-running-the-application)  
7. [API Endpoints](#-api-endpoints)  
8. [Testing](#-testing)  
9. [Project Structure](#-project-structure)  
10. [Running Frontend and Backend](#-running-frontend-and-Backend)


---
## ⚠️ Important Notice on Large Files

This repository does **not** include two large files due to GitHub’s 100MB file size limit:

1. **`Coursera_reviews.csv`**  
   - Contains over 1.4 million reviews, ratings, reviewer IDs, and course IDs.  
   - Publicly available from Kaggle: [Course Reviews on Coursera Dataset](https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera)  
   - Released under GNU GPLv2 license.
   - Download and include right next to Coursera_courses.csv in the project. 

2. **`cf_svd.pkl`**  
   - A pre-trained collaborative filtering model artifact.  
   - Not uploaded due to large size.  
   - You can regenerate it locally by running:  
```bash
     python retrain.py
```

The repository already includes pre-trained artifacts for TF-IDF and the MLP model, so only `cf_svd.pkl` require regeneration.


## Overview

This project implements a hybrid recommendation system with:

- **Content-Based Filtering (CBF):** TF-IDF on course titles+tags + cosine similarity  
- **Collaborative Filtering (CF):** Surprise KNNBasic & SVD models  
- **Hybrid Fusion:** Simple concatenation of CBF & CF results  
- **Cold-Start:** User profiling via signup and optional quiz (extendable)  
- **Explainability:** SHAP-based interpretation of MLP 
- **Web API:** FastAPI backend, protected with JWT auth  
- **Frontend:** Streamlit interface for demo  

---

## Tech Stack

- **Backend:** Python 3.10, FastAPI, SQLModel, PostgreSQL  
- **Machine Learning:** scikit-learn, Surprise, TensorFlow (for future MLP), SHAP  
- **Auth:** bcrypt, python-jose (JWT)  
- **Testing:** pytest, httpx  
- **Deployment:** Uvicorn ASGI server  

---

## Setup & Installation

1. **Clone the repo**  
```bash
   git clone <repo-url>
   cd <repo-folder>
```


2. **Create & activate virtualenv**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Configuration: 
Create a .env file in project root:

```ini
DATABASE_URL=postgresql://postgres:<PASSWORD>@localhost:5432/recommender
SECRET_KEY=<your-secret-key>
API_URL=http://127.0.0.1:8000 # Use http://<your-LAN-IP>:8000 if accessing from another device
```
>  Note: An `.env.example` file is provided.  
> Copy it to `.env` and update with your own values before running the project.



Ensure your DB is running and reachable on port 5432 
or 
follow the steps below

## Database Setup (PostgreSQL)
This project uses PostgreSQL as its database.
You must have PostgreSQL installed, running, and a database named recommender created before starting the backend.

1. **Install PostgreSQL**
Windows: Download here i.e https://www.postgresql.org/download/windows/ and follow the installer.
During installation, select PostgreSQL Server and pgAdmin 4, and you can skip Stack Builder.

macOS:
brew install postgresql

Linux (Ubuntu/Debian):
sudo apt install postgresql postgresql-contrib

2. **Start the PostgreSQL Service**
Windows (PowerShell as Admin):
net start postgresql-x64-17
(replace 17 with your installed version)

macOS/Linux:
sudo service postgresql start

3. **Create the Database**
You can choose either method below:

**Option 1 – Windows PowerShell (no PATH setup required)**
If you didn’t add PostgreSQL to your PATH:

1. Point to psql.exe (adjust if PostgreSQL is installed elsewhere)
$psql = "C:\Program Files\PostgreSQL\17\bin\psql.exe"

2. Verify psql works
& $psql --version

3. Connect to PostgreSQL (will prompt for your password)
& $psql -U postgres -h localhost -p 5432

Once connected, create the database:
CREATE DATABASE recommender;
\q

**Option 2 – If psql is in PATH (macOS, Linux, or Windows)**
psql -U postgres

Then inside the shell:
CREATE DATABASE recommender;
\q

**Option 3 – pgAdmin 4 (GUI)**

Open pgAdmin 4 → expand Servers → PostgreSQL 17 (enter your password)
Right-click Databases → Create → Database… → Name: recommender → Save

Note:
Replace postgres with your PostgreSQL username if different.
If prompted, use the password you set during installation.
To avoid "connection refused" errors, ensure the PostgreSQL service is running before connecting.

4. **Initialize tables & load data**
Run the initialization script:
```bash
python init_db.py   # creates necessary tables
python load_courses.py   # inserts courses from Coursera_courses.csv
```
Note: both `Coursera_courses.csv` and `Coursera_reviews.csv` must be present in the project root.

## Retrain Artifacts for missing cf_svd.pkl
The repo already includes pre-trained artifacts for TF-IDF, and the MLP model. Please use the command i.e retrain.py to build cf_svd.pkl since it was not uploaded to github due to large file size. Just simply run retrain.py and it will be generated in the project directory at convenience.  
If you want to regenerate them from scratch:
```bash
python retrain.py
```

## Running the Application

Start the backend:
```bash
uvicorn main:app --reload
```
Open interactive docs: http://127.0.0.1:8000/docs.

Start the frontend in new terminal:
```bash
streamlit run app.py
```
Visit: http://localhost:8501/

## API Endpoints

- **Authentication**
  - `POST /signup` → Registers a new user.  
  - `POST /login` → Returns a JWT token (use form data: `username`, `password`).  

- **Recommendations**  
  *(All recommendation endpoints require an Authorization header: `Bearer <token>`)*  
  - `GET /recommend` → Content-Based recommendations (query params: `q`, `k`).  
  - `GET /recommend/cf` → Collaborative Filtering recommendations (query params: `user_id`, `k`).  
  - `GET /recommend/hybrid` → Hybrid recommendations (query params: `user_id`, `q`, `k`).  

- **Explainability**  
  - `GET /explain/course/{course_id}` → Returns SHAP-based explainability for the MLP model.  

## Testing
Run the automated test suite:
```bash
pytest -q
```

## Project Structure 
.
├── main.py
├── auth.py
├── database.py
├── models.py
├── cbf.py
├── cf.py
├── init_db.py
├── load_courses.py
├── tests/
│   └── test_main.py
├── requirements.txt
└── README.md

## Running Frontend and Backend

1. **Start the backend**  
From the project root folder (with your virtual environment activated), run:
```bash
uvicorn main:app --reload
```
Open interactive docs: http://127.0.0.1:8000/docs.


2. **Start the frontend (in a new terminal)**
From the project root folder, run:
```bash
streamlit run app.py
```

Access in browser:
local: http://localhost:8501/
LAN: http://<your-local-ip>:8501

