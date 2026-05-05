"""
用户数据模型
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from models.database import Base


class User(Base):
    """用户模型"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=True)
    api_key = Column(String(255), unique=True, nullable=True)
    is_active = Column(Boolean, default=True)
    quota = Column(Integer, default=100)
    used_quota = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())
    last_login = Column(DateTime, nullable=True)

    def use_quota(self, amount: int = 1):
        """使用配额"""
        if self.quota - self.used_quota >= amount:
            self.used_quota += amount
        else:
            raise ValueError("配额不足")

    @property
    def remaining_quota(self) -> int:
        """剩余配额"""
        return max(0, self.quota - self.used_quota)

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"
