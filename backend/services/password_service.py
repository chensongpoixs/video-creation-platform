"""
密码加密和验证服务（直接使用 bcrypt，避免 passlib 兼容性问题）
"""
import bcrypt
import re


class PasswordService:
    """密码服务类"""

    @staticmethod
    def hash_password(password: str) -> str:
        """加密密码"""
        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """验证密码强度"""
        if len(password) < 8:
            return False, "密码长度至少 8 位"
        if not re.search(r'[A-Z]', password):
            return False, "密码必须包含至少一个大写字母"
        if not re.search(r'[a-z]', password):
            return False, "密码必须包含至少一个小写字母"
        if not re.search(r'\d', password):
            return False, "密码必须包含至少一个数字"
        return True, ""
