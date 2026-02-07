"""
用户仓储
"""
from typing import Optional
from sqlalchemy.orm import Session
from models.user import User
from repositories.base import BaseRepository

class UserRepository(BaseRepository[User]):
    """用户仓储"""
    
    def __init__(self, db: Session):
        super().__init__(User, db)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_by_api_key(self, api_key: str) -> Optional[User]:
        """根据API密钥获取"""
        return self.db.query(User).filter(User.api_key == api_key).first()
    
    def username_exists(self, username: str) -> bool:
        """检查用户名是否存在"""
        return self.db.query(User).filter(User.username == username).count() > 0
    
    def email_exists(self, email: str) -> bool:
        """检查邮箱是否存在"""
        return self.db.query(User).filter(User.email == email).count() > 0
    
    def use_quota(self, user_id: int, amount: int = 1) -> Optional[User]:
        """使用配额"""
        user = self.get(user_id)
        if user:
            user.use_quota(amount)
            self.db.flush()
            self.db.refresh(user)
        return user
    
    def reset_quota(self, user_id: int, quota: int) -> Optional[User]:
        """重置配额"""
        user = self.get(user_id)
        if user:
            user.quota = quota
            user.used_quota = 0
            self.db.flush()
            self.db.refresh(user)
        return user
