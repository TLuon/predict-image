import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
    f"?ssl_verify_cert=false"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_description(disease_name: str) -> str:
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT description FROM diseases WHERE disease_name = :name"),
            {"name": disease_name}
        ).fetchone()
        return result[0] if result else "Không có mô tả"
    except Exception:
        return "Không có mô tả"
    finally:
        db.close()