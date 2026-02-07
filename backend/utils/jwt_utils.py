"""
JWT Token 生成和验证工具
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import os


# JWT 配置
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 小时
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 天


class JWTUtils:
    """JWT 工具类"""
    
    @staticmethod
    def create_access_token(
        user_id: int,
        username: str,
        email: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建访问令牌
        
        Args:
            user_id: 用户 ID
            username: 用户名
            email: 邮箱
            expires_delta: 过期时间增量
            
        Returns:
            JWT 访问令牌
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "sub": str(user_id),
            "username": username,
            "email": email,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def create_refresh_token(
        user_id: int,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建刷新令牌
        
        Args:
            user_id: 用户 ID
            expires_delta: 过期时间增量
            
        Returns:
            JWT 刷新令牌
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        验证令牌
        
        Args:
            token: JWT 令牌
            token_type: 令牌类型 (access/refresh)
            
        Returns:
            解析后的 payload
            
        Raises:
            JWTError: 令牌无效
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # 验证令牌类型
            if payload.get("type") != token_type:
                raise JWTError(f"Invalid token type: expected {token_type}")
            
            return payload
        except JWTError as e:
            raise JWTError(f"Token verification failed: {str(e)}")
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        """
        解码令牌（不验证）
        
        Args:
            token: JWT 令牌
            
        Returns:
            解析后的 payload，失败返回 None
        """
        try:
            return jwt.decode(
                token,
                JWT_SECRET_KEY,
                algorithms=[JWT_ALGORITHM],
                options={"verify_signature": False}
            )
        except Exception:
            return None
    
    @staticmethod
    def get_token_expire_time() -> int:
        """
        获取访问令牌过期时间（秒）
        
        Returns:
            过期时间（秒）
        """
        return ACCESS_TOKEN_EXPIRE_MINUTES * 60
