import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("Falta la variable DATABASE_URL en el archivo .env")

# pool_pre_ping=True es clave para que no falle si Neon se suspende por inactividad
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True, 
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()