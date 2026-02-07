"""
密码加密和验证服务
"""
from passlib.context import CryptContext
import re


# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordService:
    """密码服务类"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        加密密码
        
        Args:
            password: 明文密码
            
        Returns:
            加密后的密码哈希
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        
        Args:
            plain_password: 明文密码
            hashed_password: 加密后的密码哈希
            
        Returns:
            密码是否匹配
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """
        验证密码强度
        
        Args:
            password: 密码
            
        Returns:
            (是否有效, 错误信息)
        """
        if len(password) < 8:
            return False, "密码长度至少 8 位"
        
        if not re.search(r'[A-Z]', password):
            return False, "密码必须包含至少一个大写字母"
        
        if not re.search(r'[a-z]', password):
            return False, "密码必须包含至少一个小写字母"
        
        if not re.search(r'\d', password):
            return False, "密码必须包含至少一个数字"
        
        return True, ""
