# models.py
from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

# ---------- Core Users & Courses ----------

class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, nullable=False, unique=True)
    email: str = Field(index=True, nullable=False, unique=True)
    hashed_password: str = Field(nullable=False)
    interests: Optional[str] = Field(
        default=None,
        description="Comma-separated list of user interest tags"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Course(SQLModel, table=True):
    __tablename__ = "courses"

    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: Optional[str] = Field(
        default=None,
        index=True,
        unique=True,
        description="Stable ID from the source CSV (e.g., course_id); optional but unique if present",
    )
    title: str = Field(nullable=False)
    description: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(
        default=None,
        description="Comma-separated list of course tags/keywords"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ---------- New tables for profiling, quiz, feedback ----------

class UserProfile(SQLModel, table=True):
    __tablename__ = "user_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, nullable=False, foreign_key="users.id")
    # free-form text fields captured from onboarding
    interests: Optional[str] = Field(default=None)
    domains: Optional[str] = Field(default=None, description="e.g., 'AI/ML, Data, Robotics'")
    goals: Optional[str] = Field(default=None, description="career goals / target roles")
    level: Optional[int] = Field(default=1, description="1=beginner, 2=intermediate, 3=advanced")
    # persisted profile vector for CBF (bytes blob; store pickled numpy array)
    profile_vector: Optional[bytes] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class QuizQuestion(SQLModel, table=True):
    __tablename__ = "quiz_questions"

    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str = Field(nullable=False)
    # optional multiple-choice options in CSV/JSON string form
    options: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuizResponse(SQLModel, table=True):
    __tablename__ = "quiz_responses"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="users.id")
    question_id: int = Field(index=True, foreign_key="quiz_questions.id")
    answer: str = Field(nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserInteraction(SQLModel, table=True):
    __tablename__ = "user_interactions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="users.id")
    course_id: str = Field(index=True, description="Matches Course.external_id if available; else Course.id as string")
    # one of: view/click/like/dislike/rating/enroll/complete
    event: str = Field(index=True)
    rating: Optional[float] = Field(default=None)  
    created_at: datetime = Field(default_factory=datetime.utcnow)
