"""Database module."""
from .models import Base, PassportRecord
from .database import get_db, init_db

__all__ = ["Base", "PassportRecord", "get_db", "init_db"]
