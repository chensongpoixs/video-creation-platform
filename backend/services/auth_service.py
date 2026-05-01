"""
认证服务
"""
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from datetime import datetime
from typing import Dict, Any

from models.user import User
from repositories.user_repository import UserRepository
from services.password_service import PasswordService
from utils.jwt_utils import JWTUtils


class AuthService:
    """认证服务类"""
    
    def __init__(self, db: Session):
        self.db = db
        self.user_repo = UserRepository(db)
    
    def register(self, username: str, email: str, password: str) -> User:
        """
        用户注册
        """
        print(f"[注册服务] 开始注册: username={username}, email={email}")

        # 检查用户名是否存在
        existing_user = self.user_repo.get_by_username(username)
        if existing_user:
            print(f"[注册服务] ❌ 用户名已存在: {username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )

        # 检查邮箱是否存在
        existing_email = self.user_repo.get_by_email(email)
        if existing_email:
            print(f"[注册服务] ❌ 邮箱已被注册: {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被注册"
            )

        # 加密密码
        password_hash = PasswordService.hash_password(password)
        print(f"[注册服务] 密码哈希: {password_hash[:30]}... (长度={len(password_hash)})")

        # 创建用户
        user_data = {
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "is_active": True,
            "quota": 100,
            "used_quota": 0
        }

        user = self.user_repo.create(**user_data)
        print(f"[注册服务] ✅ 用户创建成功: id={user.id}, username={user.username}, is_active={user.is_active}")
        return user
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        用户登录
        """
        print(f"[登录服务] 开始登录: input={username}")

        # 查找用户（支持用户名或邮箱登录）
        user = self.user_repo.get_by_username(username)
        if not user:
            print(f"[登录服务] 用户名未匹配到，尝试邮箱查询...")
            user = self.user_repo.get_by_email(username)

        if not user:
            print(f"[登录服务] ❌ 用户不存在: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误"
            )

        print(f"[登录服务] 查到用户: id={user.id}, username={user.username}, is_active={user.is_active}")
        print(f"[登录服务] DB中password_hash: {user.password_hash[:30] if user.password_hash else 'None'}...")

        # 检查用户是否激活
        if not user.is_active:
            print(f"[登录服务] ❌ 用户已被禁用")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="用户已被禁用"
            )

        # 验证密码
        if not user.password_hash:
            print(f"[登录服务] ❌ 用户无密码哈希")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户未设置密码，请使用其他方式登录"
            )

        verify_result = PasswordService.verify_password(password, user.password_hash)
        print(f"[登录服务] 密码验证结果: {verify_result}")

        if not verify_result:
            print(f"[登录服务] ❌ 密码不匹配")
            # 额外调试：用相同输入的密码重新哈希看是否一致
            rehash_test = PasswordService.hash_password(password)
            print(f"[登录服务] 调试: 重新hash输入密码 = {rehash_test[:30]}...")
            print(f"[登录服务] 调试: DB存储的hash    = {user.password_hash[:30]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误"
            )

        print(f"[登录服务] ✅ 密码验证通过")
        
        # 更新最后登录时间
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        # 生成 Token
        access_token = JWTUtils.create_access_token(
            user_id=user.id,
            username=user.username,
            email=user.email
        )
        
        refresh_token = JWTUtils.create_refresh_token(user_id=user.id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": JWTUtils.get_token_expire_time()
        }
    
    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        刷新访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            新的 Token 字典
            
        Raises:
            HTTPException: 令牌无效
        """
        try:
            # 验证刷新令牌
            payload = JWTUtils.verify_token(refresh_token, token_type="refresh")
            user_id = int(payload.get("sub"))
            
            # 获取用户
            user = self.user_repo.get(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="用户不存在"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="用户已被禁用"
                )
            
            # 生成新的访问令牌
            access_token = JWTUtils.create_access_token(
                user_id=user.id,
                username=user.username,
                email=user.email
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": JWTUtils.get_token_expire_time()
            }
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"令牌无效: {str(e)}"
            )
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """
        修改密码
        
        Args:
            user_id: 用户 ID
            old_password: 旧密码
            new_password: 新密码
            
        Returns:
            是否成功
            
        Raises:
            HTTPException: 旧密码错误
        """
        # 获取用户
        user = self.user_repo.get(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 验证旧密码
        if not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户未设置密码"
            )
        
        if not PasswordService.verify_password(old_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="旧密码错误"
            )
        
        # 加密新密码
        new_password_hash = PasswordService.hash_password(new_password)
        
        # 更新密码
        user.password_hash = new_password_hash
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True
    
    def get_current_user(self, token: str) -> User:
        """
        根据 Token 获取当前用户
        
        Args:
            token: 访问令牌
            
        Returns:
            用户对象
            
        Raises:
            HTTPException: 令牌无效或用户不存在
        """
        try:
            # 验证令牌
            payload = JWTUtils.verify_token(token, token_type="access")
            user_id = int(payload.get("sub"))
            
            # 获取用户
            user = self.user_repo.get(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="用户不存在"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="用户已被禁用"
                )
            
            return user
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"认证失败: {str(e)}"
            )
