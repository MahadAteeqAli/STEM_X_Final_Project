from database import create_db_and_tables
import models  # ensures models are registered

if __name__ == "__main__":
    create_db_and_tables()
    print("✅ Database tables created/verified.")
