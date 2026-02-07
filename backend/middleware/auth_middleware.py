"""
认证中间件和依赖
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from models.database import get_db
from models.user import User
from services.auth_service import AuthService


# HTTP Bearer 认证方案
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    获取当前认证用户（依赖注入）
    
    Args:
        credentials: HTTP 认证凭证
        db: 数据库会话
        
    Returns:
        当前用户对象
        
    Raises:
        HTTPException: 认证失败
    """
    token = credentials.credentials
    auth_service = AuthService(db)
    return auth_service.get_current_user(token)


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前激活用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        当前激活用户
        
    Raises:
        HTTPException: 用户未激活
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    获取可选的当前用户（不强制认证）
    
    Args:
        credentials: HTTP 认证凭证（可选）
        db: 数据库会话
        
    Returns:
        当前用户对象或 None
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        auth_service = AuthService(db)
        return auth_service.get_current_user(token)
    except Exception:
        return None
