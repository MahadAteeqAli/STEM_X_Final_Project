import pandas as pd
from sqlmodel import Session, select
from database import engine
from models import Course

CSV = "Coursera_courses.csv"

if __name__ == "__main__":
    df = pd.read_csv(CSV)

    # Ensure "tags" column exists 
    if "tags" not in df.columns:
        df["tags"] = ""
    else:
        df["tags"] = df["tags"].fillna("")

    with Session(engine) as s:
        cnt = 0
        for _, r in df.iterrows():
            cid = str(r["course_id"])
            title = str(r["name"])  
            tags = str(r.get("tags", ""))

            exists = s.exec(
                select(Course).where(Course.external_id == cid)
            ).first()

            if not exists:
                s.add(Course(external_id=cid, title=title, tags=tags))
                cnt += 1

        s.commit()
        print(f"✅ Upserted {cnt} courses.")
