"""
任务数据模型
"""
from enum import Enum
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from models.database import Base


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(Base):
    """任务模型"""

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    prompt = Column(Text, nullable=True)
    status = Column(String(20), default=TaskStatus.PENDING.value, index=True)
    progress = Column(Integer, default=0)
    completed_scenes = Column(Integer, nullable=True)
    total_scenes = Column(Integer, default=0)
    duration = Column(Float, nullable=True)
    video_path = Column(String(512), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())

    # 关系
    scripts = relationship("Script", back_populates="task", lazy="select")
    videos = relationship("Video", back_populates="task", lazy="select")

    def start(self):
        """开始任务"""
        self.status = TaskStatus.PROCESSING.value
        self.progress = 5  # 初始进度
        self.updated_at = datetime.utcnow()

    def complete(self, video_path: Optional[str] = None):
        """完成任务"""
        self.status = TaskStatus.COMPLETED.value
        self.progress = 100
        if video_path:
            self.video_path = video_path
        self.updated_at = datetime.utcnow()

    def fail(self, error_message: str):
        """任务失败"""
        self.status = TaskStatus.FAILED.value
        self.error_message = error_message
        self.updated_at = datetime.utcnow()

    def __repr__(self):
        return f"<Task(task_id={self.task_id}, status={self.status})>"
