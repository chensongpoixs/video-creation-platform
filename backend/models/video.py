"""
视频资源数据模型
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from models.database import Base


class Video(Base):
    """视频资源模型"""

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    scene_number = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    # 关系
    task = relationship("Task", back_populates="videos")

    def __repr__(self):
        return f"<Video(id={self.id}, task_id={self.task_id}, scene={self.scene_number})>"
