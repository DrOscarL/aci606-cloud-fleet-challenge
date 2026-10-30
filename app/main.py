from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
import time

# Importamos nuestra configuración de base de datos y el modelo
from .db import SessionLocal, engine, Base
from .models import Telemetry

# Creamos las tablas en Neon si no existen
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ACI606 - Cloud Fleet Challenge")

# Dependencia para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- MODELOS PYDANTIC (Contratos de entrada y salida) ---
class TelemetryIn(BaseModel):
    robot_id: str
    timestamp: float
    x: float
    y: float
    speed: float
    battery: float
    motor_temp: float
    vibration: float
    motor_current: float
    obstacle_distance: float

# --- ENDPOINTS ---

@app.get("/health")
def health():
    return {"status": "ok", "service": "running"}

@app.post("/telemetry")
def post_telemetry(payload: TelemetryIn, db: Session = Depends(get_db)):
    # Guardamos en PostgreSQL
    new_record = Telemetry(**payload.model_dump())
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"status": "ok", "id": new_record.id}

@app.get("/robots")
def get_robots(db: Session = Depends(get_db)):
    # Obtiene la última telemetría de cada robot
    subquery = db.query(
        Telemetry.robot_id, 
        func.max(Telemetry.timestamp).label("max_ts")
    ).group_by(Telemetry.robot_id).subquery()

    results = db.query(Telemetry).join(
        subquery,
        (Telemetry.robot_id == subquery.c.robot_id) & 
        (Telemetry.timestamp == subquery.c.max_ts)
    ).all()
    
    return results

@app.get("/robots/{robot_id}/history")
def get_history(robot_id: str, db: Session = Depends(get_db)):
    # Retorna todo el historial de un robot específico
    results = db.query(Telemetry).filter(
        Telemetry.robot_id == robot_id
    ).order_by(Telemetry.timestamp.asc()).all()
    
    if not results:
        raise HTTPException(status_code=404, detail="Robot no encontrado")
        
    return results