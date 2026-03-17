import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from core.config import settings

connect_args = {"check_same_thread": False} if settings.is_sqlite else {}
engine = create_async_engine(
    settings.async_database_url,
    echo=False,
    connect_args=connect_args,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class Repository(Base):
    __tablename__ = "repositories"
    id               = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name             = Column(String(255), nullable=False)
    source           = Column(String(20), nullable=False)
    url              = Column(String(1024), nullable=True)
    branch           = Column(String(255), default="main")
    status           = Column(String(20), default="pending")
    error_message    = Column(Text, nullable=True)
    language         = Column(String(64), nullable=True)
    languages        = Column(JSON, default=dict)
    tech_stack       = Column(JSON, default=list)
    total_files      = Column(Integer, default=0)
    total_lines      = Column(Integer, default=0)
    file_tree        = Column(JSON, nullable=True)
    architecture     = Column(Text, nullable=True)
    vector_namespace = Column(String(255), nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    analyzed_at      = Column(DateTime, nullable=True)
    files         = relationship("RepoFile",    back_populates="repository", cascade="all, delete-orphan")
    analyses      = relationship("Analysis",    back_populates="repository", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="repository", cascade="all, delete-orphan")


class RepoFile(Base):
    __tablename__ = "repo_files"
    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    repo_id    = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    path       = Column(String(2048), nullable=False)
    name       = Column(String(512), nullable=False)
    extension  = Column(String(32), nullable=True)
    language   = Column(String(64), nullable=True)
    size_bytes = Column(Integer, default=0)
    line_count = Column(Integer, default=0)
    content    = Column(Text, nullable=True)
    summary    = Column(Text, nullable=True)
    functions  = Column(JSON, default=list)
    classes    = Column(JSON, default=list)
    imports    = Column(JSON, default=list)
    is_binary  = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    repository = relationship("Repository", back_populates="files")


class Analysis(Base):
    __tablename__ = "analyses"
    id          = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    repo_id     = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    type        = Column(String(32), nullable=False)
    target_path = Column(String(2048), nullable=True)
    title       = Column(String(512), nullable=True)
    content     = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)
    repository  = relationship("Repository", back_populates="analyses")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    repo_id    = Column(String(36), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    title      = Column(String(512), default="New conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    repository = relationship("Repository", back_populates="chat_sessions")
    messages   = relationship("ChatMessage", back_populates="session",
                              cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(16), nullable=False)
    content    = Column(Text, nullable=False)
    sources    = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    session    = relationship("ChatSession", back_populates="messages")


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
