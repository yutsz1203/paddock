"""Engine and session factory.

One engine per process, created lazily so that importing this module does not
require a reachable database — tests that never touch Postgres stay fast.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from paddock.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(str(get_settings().database_url), pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception.

    Ingestion relies on the rollback: a meeting that fails part-way must leave no
    partial meeting behind (plan T10).
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
