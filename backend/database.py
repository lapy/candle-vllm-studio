import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.logging_config import get_logger

logger = get_logger(__name__)

Base = declarative_base()

_engine = None
_session_factory = None


def resolve_data_dir() -> str:
    """Return the configured data directory (without creating it)."""
    return os.getenv("CANDLE_STUDIO_DATA", os.path.join(os.getcwd(), "data"))


def _ensure_session_factory():
    global _engine, _session_factory
    if _session_factory is None:
        data_dir = Path(resolve_data_dir())
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "db.sqlite"
        database_url = f"sqlite:///{db_path}"
        _engine = create_engine(database_url, connect_args={"check_same_thread": False})
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        logger.info("Database engine initialised at %s", db_path)
    return _session_factory


def get_engine():
    _ensure_session_factory()
    return _engine


def SessionLocal():
    """Return a new SQLAlchemy session."""
    factory = _ensure_session_factory()
    return factory()


def get_db():
    """FastAPI dependency for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_runtime_alias(huggingface_id: str, quantization: Optional[str] = None) -> str:
    """Generate a stable alias for referencing model variants."""
    slug = huggingface_id.replace("/", "-").replace(" ", "-").replace(".", "-").lower()
    if quantization:
        quant_slug = quantization.replace(" ", "-").lower()
        return f"{slug}.{quant_slug}"
    return slug


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    huggingface_id = Column(String, index=True)
    base_model_name = Column(String, index=True)
    file_path = Column(String)
    file_size = Column(Integer)
    quantization = Column(String)
    model_type = Column(String)
    downloaded_at = Column(DateTime)
    is_active = Column(Boolean, default=False)
    config = Column(JSON)  # candle runtime configuration
    runtime_alias = Column(String, index=True)


class CandleBuild(Base):
    __tablename__ = "candle_builds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    git_ref = Column(String)
    binary_path = Column(String)
    features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    build_metadata = Column("metadata", JSON)


class RunningInstance(Base):
    __tablename__ = "running_instances"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, index=True)
    build_name = Column(String, nullable=True)
    port = Column(Integer)
    pid = Column(Integer, nullable=True)
    endpoint = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    config = Column(JSON)


def sync_model_active_status(db):
    """Ensure model.is_active reflects running instances state."""
    running = db.query(RunningInstance).all()
    active_ids = {instance.model_id for instance in running}

    updated = 0
    for model in db.query(Model).all():
        new_status = model.id in active_ids
        if model.is_active != new_status:
            model.is_active = new_status
            updated += 1

    if updated:
        db.commit()
        logger.info("Synced %s models' is_active status", updated)

    return updated


async def init_db():
    """Initialise database tables and run lightweight migrations."""
    data_dir = Path(resolve_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=get_engine())
    _ensure_schema_evolution()
    migrate_existing_models()


def _ensure_schema_evolution() -> None:
    """Ensure new columns exist on legacy SQLite databases."""
    with get_engine().begin() as conn:
        def column_missing(table: str, column: str) -> bool:
            result = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            return column not in {row[1] for row in result}

        if column_missing("models", "runtime_alias"):
            conn.execute(text("ALTER TABLE models ADD COLUMN runtime_alias TEXT"))
            logger.info("Added runtime_alias column to models table")

        if column_missing("running_instances", "build_name"):
            conn.execute(text("ALTER TABLE running_instances ADD COLUMN build_name TEXT"))
        if column_missing("running_instances", "pid"):
            conn.execute(text("ALTER TABLE running_instances ADD COLUMN pid INTEGER"))
        if column_missing("running_instances", "endpoint"):
            conn.execute(text("ALTER TABLE running_instances ADD COLUMN endpoint TEXT"))


def migrate_existing_models():
    """Populate base model name/runtime alias for legacy records."""
    db = SessionLocal()
    try:
        models = db.query(Model).all()
        patched = 0

        for model in models:
            if not model.base_model_name:
                if model.huggingface_id:
                    parts = model.huggingface_id.split("/")
                    model.base_model_name = parts[-1] if parts else model.huggingface_id
                elif model.name:
                    model.base_model_name = model.name.split("-")[0]
                else:
                    model.base_model_name = "unknown"
                patched += 1

            if isinstance(model.config, str):
                try:
                    model.config = json.loads(model.config)
                except json.JSONDecodeError:
                    model.config = None

            if not model.runtime_alias and model.huggingface_id:
                model.runtime_alias = generate_runtime_alias(model.huggingface_id, model.quantization)
                patched += 1

        if patched:
            db.commit()
            logger.info("Migrated %s models to populate base_model_name/runtime_alias", patched)
    except Exception as exc:
        logger.error("Error migrating models: %s", exc)
        db.rollback()
    finally:
        db.close()
