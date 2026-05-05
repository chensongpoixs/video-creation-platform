"""
数据库配置和会话管理

SQLite 并发策略:
- timeout=30: 在 sqlite3.connect() 时传入，由 C 级 SQLite 库处理忙等待
  这比 busy_timeout PRAGMA 更可靠（后者在某些情况下可能不触发）
- NullPool: 禁用连接池，每会话独立连接，用完即销毁，杜绝跨线程锁残留
- WAL 模式: 写不阻塞读，读不阻塞写
- busy_timeout: 双重保险（10 秒）
"""
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from config import DATABASE_URL

# SQLAlchemy 声明式基类
Base = declarative_base()

if "sqlite" in DATABASE_URL:
    # SQLite: NullPool + timeout 防锁
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,  # ← C 级忙等待 30 秒，最可靠的防锁手段
        },
        poolclass=NullPool,
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")  # 双重保险
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
else:
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
    )

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI 依赖注入：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context():
    """上下文管理器：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """初始化数据库：创建所有表"""
    from models.user import User  # noqa: F401
    from models.task import Task  # noqa: F401
    from models.video import Video  # noqa: F401
    from models.script import Script  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db_info() -> dict:
    """获取数据库信息"""
    import os
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    info = {
        "url": DATABASE_URL,
        "tables": tables,
    }

    if "sqlite" in DATABASE_URL:
        db_path = DATABASE_URL.replace("sqlite:///", "")
        if os.path.exists(db_path):
            info["size_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 2)

    return info
