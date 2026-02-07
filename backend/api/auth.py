"""
认证 API 路由
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from models.database import get_db
from models.user import User
from schemas.auth import (
    RegisterSchema,
    LoginSchema,
    TokenSchema,
    RefreshTokenSchema,
    ChangePasswordSchema,
    UserResponseSchema
)
from services.auth_service import AuthService
from middleware.auth_middleware import get_current_active_user


router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post("/register", response_model=dict, summary="用户注册")
async def register(
    data: RegisterSchema,
    db: Session = Depends(get_db)
):
    """
    用户注册
    
    - **username**: 用户名（3-50字符，只能包含字母、数字和下划线）
    - **email**: 邮箱
    - **password**: 密码（至少8位，包含大小写字母和数字）
    """
    auth_service = AuthService(db)
    user = auth_service.register(
        username=data.username,
        email=data.email,
        password=data.password
    )
    
    return {
        "message": "注册成功",
        "user_id": user.id,
        "username": user.username
    }


@router.post("/login", response_model=TokenSchema, summary="用户登录")
async def login(
    data: LoginSchema,
    db: Session = Depends(get_db)
):
    """
    用户登录
    
    - **username**: 用户名或邮箱
    - **password**: 密码
    
    返回访问令牌和刷新令牌
    """
    auth_service = AuthService(db)
    tokens = auth_service.login(
        username=data.username,
        password=data.password
    )
    
    return tokens


@router.post("/refresh", response_model=dict, summary="刷新令牌")
async def refresh_token(
    data: RefreshTokenSchema,
    db: Session = Depends(get_db)
):
    """
    刷新访问令牌
    
    - **refresh_token**: 刷新令牌
    
    返回新的访问令牌
    """
    auth_service = AuthService(db)
    tokens = auth_service.refresh_token(data.refresh_token)
    
    return tokens


@router.get("/me", response_model=UserResponseSchema, summary="获取当前用户信息")
async def get_me(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取当前登录用户的信息
    
    需要在请求头中携带访问令牌：
    ```
    Authorization: Bearer <access_token>
    ```
    """
    return UserResponseSchema(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        quota=current_user.quota,
        used_quota=current_user.used_quota,
        remaining_quota=current_user.quota - current_user.used_quota,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.post("/change-password", response_model=dict, summary="修改密码")
async def change_password(
    data: ChangePasswordSchema,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    修改密码
    
    - **old_password**: 旧密码
    - **new_password**: 新密码（至少8位，包含大小写字母和数字）
    
    需要认证
    """
    auth_service = AuthService(db)
    auth_service.change_password(
        user_id=current_user.id,
        old_password=data.old_password,
        new_password=data.new_password
    )
    
    return {"message": "密码修改成功"}


@router.post("/logout", response_model=dict, summary="用户登出")
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """
    用户登出
    
    注意：由于使用 JWT，服务端无法主动撤销令牌。
    客户端应该删除本地存储的令牌。
    """
    return {"message": "登出成功"}
