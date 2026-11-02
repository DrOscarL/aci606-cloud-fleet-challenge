
# ==========================================
# ACI606 - CLOUD FLEET CHALLENGE
# Conexion y persistencia PostgreSQL
# ==========================================

import os
from pathlib import Path

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    String,
    Float,
    Integer,
    Index
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker
)


# ==========================================
# 1. CARGAR VARIABLES DE ENTORNO
# ==========================================

# Obtener la carpeta principal del proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Localizar el archivo .env
ENV_FILE = ROOT_DIR / ".env"

# Cargar las variables
load_dotenv(dotenv_path=ENV_FILE)

# Obtener la URL de Neon
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        f"No se encontro DATABASE_URL. "
        f"Revisa el archivo: {ENV_FILE}"
    )


# ==========================================
# 2. ADAPTAR LA CONEXION A PSYCOPG 3
# ==========================================

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )


# ==========================================
# 3. CREAR MOTOR DE CONEXION
# ==========================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ==========================================
# 4. CREAR ADMINISTRADOR DE SESIONES
# ==========================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)


# ==========================================
# 5. DEFINIR BASE DE MODELOS
# ==========================================

class Base(DeclarativeBase):
    pass


# ==========================================
# 6. DEFINIR TABLA DE TELEMETRIA
# ==========================================

class TelemetryRecord(Base):

    __tablename__ = "telemetry"

    # Identificador unico del registro
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Identificador del robot
    robot_id: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True
    )

    # Mediciones del robot
    timestamp: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    x: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    y: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    speed: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    battery: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    motor_temp: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    vibration: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    motor_current: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    obstacle_distance: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    # Convertir el registro a diccionario
    def to_dict(self):

        return {
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "battery": self.battery,
            "motor_temp": self.motor_temp,
            "vibration": self.vibration,
            "motor_current": self.motor_current,
            "obstacle_distance": self.obstacle_distance
        }


# ==========================================
# 7. CREAR TABLAS EN POSTGRESQL
# ==========================================

def init_db():

    Base.metadata.create_all(bind=engine)

    print("Base de datos inicializada correctamente.")
