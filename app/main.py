from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import DATABASE_URL, TelemetryRow, engine, get_session, init_db

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # crea las tablas si no existen
    yield


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.2.0-persistence",
    description="Cloud Fleet Challenge API with persistent storage.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Responde 422 sin devolver el valor recibido.

    Por defecto FastAPI incluye el `input` inválido en la respuesta; si es NaN,
    json.dumps falla y el servidor termina respondiendo 500. Aquí lo omitimos.
    """
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class Telemetry(BaseModel):
    # NaN e Infinity se rechazan con 422 (nunca llegan a la base de datos).
    model_config = ConfigDict(allow_inf_nan=False)

    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)


class PredictionRequest(Telemetry):
    pass


# Contador de proceso (se reinicia con el servicio; en la etapa de /metrics
# se complementa con datos de la base de datos).
REQUEST_COUNT = 0
STARTED_AT = datetime.now(timezone.utc)


@app.get("/health")
def health() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry, db: Session = Depends(get_session)) -> dict:
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    db.add(TelemetryRow(**sample.model_dump()))
    db.commit()
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/robots")
def list_robots(db: Session = Depends(get_session)) -> list[dict]:
    """Última muestra de cada robot."""
    last_ids = select(func.max(TelemetryRow.id)).group_by(TelemetryRow.robot_id)
    rows = db.scalars(
        select(TelemetryRow).where(TelemetryRow.id.in_(last_ids)).order_by(TelemetryRow.robot_id)
    ).all()
    return [r.to_dict() for r in rows]


@app.get("/robots/{robot_id}/history")
def robot_history(
    robot_id: str,
    limit: int | None = Query(default=None, ge=1, le=100_000),
    db: Session = Depends(get_session),
) -> list[dict]:
    """Historial completo (orden cronológico). `limit` opcional devuelve las últimas N."""
    exists = db.scalar(select(TelemetryRow.id).where(TelemetryRow.robot_id == robot_id).limit(1))
    if exists is None:
        raise HTTPException(status_code=404, detail="Robot not found")

    stmt = select(TelemetryRow).where(TelemetryRow.robot_id == robot_id)
    if limit is None:
        rows = db.scalars(stmt.order_by(TelemetryRow.id)).all()
    else:
        rows = list(reversed(db.scalars(stmt.order_by(TelemetryRow.id.desc()).limit(limit)).all()))
    return [r.to_dict() for r in rows]


@app.post("/predict")
def predict(_: PredictionRequest) -> dict:
    # TODO E4: load the trained model and return a real prediction.
    raise HTTPException(
        status_code=501,
        detail="/predict is part of the challenge. Train and integrate your model.",
    )


@app.get("/metrics")
def metrics(db: Session = Depends(get_session)) -> dict:
    # TODO E4: extend with latency, errors, throughput and model metrics.
    total = db.scalar(select(func.count(TelemetryRow.id))) or 0
    robots = db.scalar(select(func.count(func.distinct(TelemetryRow.robot_id)))) or 0
    return {
        "requests_received": REQUEST_COUNT,
        "telemetry_rows_total": total,
        "robots_total": robots,
        "persistence": "SQLITE" if DATABASE_URL.startswith("sqlite") else "POSTGRESQL",
        "uptime_seconds": round((datetime.now(timezone.utc) - STARTED_AT).total_seconds(), 1),
        "prediction_service": "NOT_IMPLEMENTED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
