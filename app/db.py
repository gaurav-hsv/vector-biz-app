from psycopg_pool import ConnectionPool
from typing import Optional
from .config import settings

_pool: Optional[ConnectionPool] = None

def init_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo=settings.PG_DSN, min_size=1, max_size=10, open=True)
    return _pool

def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_pool() on startup.")
    return _pool

def close_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None

def db_ready() -> bool:
    pool = get_pool()
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
        return True
    except Exception:
        return False
