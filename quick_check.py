# quick_check.py
from sqlmodel import Session, select
from database import engine
from models import Course

with Session(engine) as session:
    courses = session.exec(select(Course)).all()
    print(f"Total courses in DB: {len(courses)}")
    print(courses[:3])  
