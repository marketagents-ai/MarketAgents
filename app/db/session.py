from sqlmodel import create_engine, SQLModel, Session
from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URI,
    echo=False,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

def get_session() -> Session:
    return Session(engine)