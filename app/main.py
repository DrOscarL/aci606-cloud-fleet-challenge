from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
import time
import joblib
import pandas as pd
from pathlib import Path

# Importamos nuestra configuración de base de datos y el modelo
from .db import SessionLocal, engine, Base
from .models import Telemetry

# Creamos las tablas en Neon si no existen
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ACI606 - Cloud Fleet Challenge")

# --- CARGAR EL MODELO ML ---
MODEL_PATH = Path("models/model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError("Modelo no encontrado. Ejecuta python ml/train_model.py")

# Se carga una sola vez al iniciar el servidor
model = joblib.load(MODEL_PATH)
MODEL_VERSION = "v1"

# Las mismas features con las que entrenaste el modelo
FEATURES = [
    "speed", "battery", "motor_temp", 
    "vibration", "motor_current", "obstacle_distance"
]

# Dependencia para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- MODELOS PYDANTIC ---
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

class PredictIn(BaseModel):
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
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/telemetry")
def post_telemetry(payload: TelemetryIn, db: Session = Depends(get_db)):
    new_record = Telemetry(**payload.model_dump())
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"status": "ok", "id": new_record.id}

@app.get("/robots")
def get_robots(db: Session = Depends(get_db)):
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
    results = db.query(Telemetry).filter(
        Telemetry.robot_id == robot_id
    ).order_by(Telemetry.timestamp.asc()).all()
    if not results:
        raise HTTPException(status_code=404, detail="Robot no encontrado")
    return results

@app.post("/predict")
def predict(payload: PredictIn):
    start_time = time.perf_counter()
    
    # 1. Extraer solo las features que el modelo necesita
    input_data = {k: getattr(payload, k) for k in FEATURES}
    
    # 2. Convertir a DataFrame (el modelo fue entrenado con pandas)
    X = pd.DataFrame([input_data])
    
    # 3. Predecir
    prediction = model.predict(X)[0]
    
    # 4. Obtener la confianza (probabilidad de la clase predicha)
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))
    
    # 5. Calcular tiempo de inferencia
    end_time = time.perf_counter()
    inference_ms = round((end_time - start_time) * 1000, 2)
    
    # 6. Retornar el contrato exacto
    return {
        "robot_id": payload.robot_id,
        "predicted_status": str(prediction),
        "confidence": round(confidence, 4),
        "model_version": MODEL_VERSION,
        "inference_ms": inference_ms
    }