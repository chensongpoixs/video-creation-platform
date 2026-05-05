"""
分镜脚本数据模型
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from models.database import Base


class Script(Base):
    """分镜脚本模型"""

    __tablename__ = "scripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    scene_number = Column(Integer, nullable=False)
    description = Column(String(1024), nullable=True)
    duration = Column(Float, default=5.0)
    camera_movement = Column(String(100), nullable=True)
    lighting = Column(String(100), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    # 关系
    task = relationship("Task", back_populates="scripts")

    def __repr__(self):
        return f"<Script(id={self.id}, task_id={self.task_id}, scene={self.scene_number})>"
