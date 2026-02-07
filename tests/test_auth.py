"""
认证系统测试
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.models.database import Base, get_db
from backend.models.user import User
from backend.services.password_service import PasswordService


# 测试数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_auth.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """覆盖数据库依赖"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """每个测试前创建数据库，测试后删除"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


class TestUserRegistration:
    """用户注册测试"""
    
    def test_register_success(self):
        """测试成功注册"""
        response = client.post("/api/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test1234"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "注册成功"
        assert data["username"] == "testuser"
        assert "user_id" in data
    
    def test_register_duplicate_username(self):
        """测试重复用户名"""
        # 第一次注册
        client.post("/api/auth/register", json={
            "username": "testuser",
            "email": "test1@example.com",
            "password": "Test1234"
        })
        
        # 第二次注册相同用户名
        response = client.post("/api/auth/register", json={
            "username": "testuser",
            "email": "test2@example.com",
            "password": "Test1234"
        })
        
        assert response.status_code == 400
        assert "用户名已存在" in response.json()["detail"]
    
    def test_register_duplicate_email(self):
        """测试重复邮箱"""
        # 第一次注册
        client.post("/api/auth/register", json={
            "username": "testuser1",
            "email": "test@example.com",
            "password": "Test1234"
        })
        
        # 第二次注册相同邮箱
        response = client.post("/api/auth/register", json={
            "username": "testuser2",
            "email": "test@example.com",
            "password": "Test1234"
        })
        
        assert response.status_code == 400
        assert "邮箱已被注册" in response.json()["detail"]
    
    def test_register_invalid_username(self):
        """测试无效用户名"""
        response = client.post("/api/auth/register", json={
            "username": "test user",  # 包含空格
            "email": "test@example.com",
            "password": "Test1234"
        })
        
        assert response.status_code == 422
    
    def test_register_weak_password(self):
        """测试弱密码"""
        response = client.post("/api/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "weak"  # 太短
        })
        
        assert response.status_code == 422


class TestUserLogin:
    """用户登录测试"""
    
    def setup_method(self):
        """每个测试前创建测试用户"""
        client.post("/api/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test1234"
        })
    
    def test_login_success(self):
        """测试成功登录"""
        response = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "Test1234"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_with_email(self):
        """测试使用邮箱登录"""
        response = client.post("/api/auth/login", json={
            "username": "test@example.com",
            "password": "Test1234"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
    
    def test_login_wrong_password(self):
        """测试错误密码"""
        response = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "WrongPass123"
        })
        
        assert response.status_code == 401
        assert "用户名或密码错误" in response.json()["detail"]
    
    def test_login_nonexistent_user(self):
        """测试不存在的用户"""
        response = client.post("/api/auth/login", json={
            "username": "nonexistent",
            "password": "Test1234"
        })
        
        assert response.status_code == 401
        assert "用户名或密码错误" in response.json()["detail"]


class TestTokenOperations:
    """Token 操作测试"""
    
    def setup_method(self):
        """每个测试前创建测试用户并登录"""
        client.post("/api/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test1234"
        })
        
        response = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "Test1234"
        })
        
        self.tokens = response.json()
        self.access_token = self.tokens["access_token"]
        self.refresh_token = self.tokens["refresh_token"]
    
    def test_get_current_user(self):
        """测试获取当前用户信息"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert "quota" in data
        assert "remaining_quota" in data
    
    def test_get_current_user_without_token(self):
        """测试未携带 Token"""
        response = client.get("/api/auth/me")
        
        assert response.status_code == 403
    
    def test_get_current_user_invalid_token(self):
        """测试无效 Token"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401
    
    def test_refresh_token(self):
        """测试刷新 Token"""
        response = client.post("/api/auth/refresh", json={
            "refresh_token": self.refresh_token
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_refresh_token_invalid(self):
        """测试无效刷新 Token"""
        response = client.post("/api/auth/refresh", json={
            "refresh_token": "invalid_refresh_token"
        })
        
        assert response.status_code == 401


class TestPasswordOperations:
    """密码操作测试"""
    
    def setup_method(self):
        """每个测试前创建测试用户并登录"""
        client.post("/api/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test1234"
        })
        
        response = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "Test1234"
        })
        
        self.access_token = response.json()["access_token"]
    
    def test_change_password_success(self):
        """测试成功修改密码"""
        response = client.post(
            "/api/auth/change-password",
            json={
                "old_password": "Test1234",
                "new_password": "NewPass1234"
            },
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        
        assert response.status_code == 200
        assert response.json()["message"] == "密码修改成功"
        
        # 验证新密码可以登录
        login_response = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "NewPass1234"
        })
        assert login_response.status_code == 200
    
    def test_change_password_wrong_old_password(self):
        """测试旧密码错误"""
        response = client.post(
            "/api/auth/change-password",
            json={
                "old_password": "WrongPass123",
                "new_password": "NewPass1234"
            },
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        
        assert response.status_code == 401
        assert "旧密码错误" in response.json()["detail"]
    
    def test_change_password_weak_new_password(self):
        """测试新密码太弱"""
        response = client.post(
            "/api/auth/change-password",
            json={
                "old_password": "Test1234",
                "new_password": "weak"
            },
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        
        assert response.status_code == 422


class TestPasswordService:
    """密码服务测试"""
    
    def test_hash_password(self):
        """测试密码加密"""
        password = "Test1234"
        hashed = PasswordService.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password_correct(self):
        """测试验证正确密码"""
        password = "Test1234"
        hashed = PasswordService.hash_password(password)
        
        assert PasswordService.verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """测试验证错误密码"""
        password = "Test1234"
        hashed = PasswordService.hash_password(password)
        
        assert PasswordService.verify_password("WrongPass", hashed) is False
    
    def test_validate_password_strength(self):
        """测试密码强度验证"""
        # 有效密码
        valid, msg = PasswordService.validate_password_strength("Test1234")
        assert valid is True
        
        # 太短
        valid, msg = PasswordService.validate_password_strength("Test12")
        assert valid is False
        assert "至少 8 位" in msg
        
        # 缺少大写字母
        valid, msg = PasswordService.validate_password_strength("test1234")
        assert valid is False
        assert "大写字母" in msg
        
        # 缺少小写字母
        valid, msg = PasswordService.validate_password_strength("TEST1234")
        assert valid is False
        assert "小写字母" in msg
        
        # 缺少数字
        valid, msg = PasswordService.validate_password_strength("TestTest")
        assert valid is False
        assert "数字" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
