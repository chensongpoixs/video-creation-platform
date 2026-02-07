"""
认证系统快速测试脚本
"""
import sys
import os

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.password_service import PasswordService
from utils.jwt_utils import JWTUtils
from utils.logger import setup_logger

logger = setup_logger(__name__)

def test_password_service():
    """测试密码服务"""
    print("\n" + "="*60)
    print("测试密码服务")
    print("="*60)
    
    try:
        # 测试密码加密
        password = "Test1234"
        print(f"\n1. 加密密码: {password}")
        hashed = PasswordService.hash_password(password)
        print(f"   加密结果: {hashed[:50]}...")
        
        # 测试密码验证（正确）
        print(f"\n2. 验证正确密码")
        result = PasswordService.verify_password(password, hashed)
        print(f"   验证结果: {result}")
        assert result is True, "密码验证失败"
        
        # 测试密码验证（错误）
        print(f"\n3. 验证错误密码")
        result = PasswordService.verify_password("WrongPass", hashed)
        print(f"   验证结果: {result}")
        assert result is False, "应该验证失败"
        
        # 测试密码强度验证
        print(f"\n4. 测试密码强度验证")
        test_cases = [
            ("Test1234", True, "有效密码"),
            ("weak", False, "太短"),
            ("test1234", False, "缺少大写字母"),
            ("TEST1234", False, "缺少小写字母"),
            ("TestTest", False, "缺少数字"),
        ]
        
        for pwd, expected, desc in test_cases:
            valid, msg = PasswordService.validate_password_strength(pwd)
            status = "✅" if valid == expected else "❌"
            print(f"   {status} {desc}: {pwd} -> {valid} ({msg})")
        
        print("\n✅ 密码服务测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 密码服务测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_jwt_utils():
    """测试 JWT 工具"""
    print("\n" + "="*60)
    print("测试 JWT 工具")
    print("="*60)
    
    try:
        # 测试创建访问令牌
        print(f"\n1. 创建访问令牌")
        user_id = 1
        username = "testuser"
        email = "test@example.com"
        
        access_token = JWTUtils.create_access_token(user_id, username, email)
        print(f"   访问令牌: {access_token[:50]}...")
        
        # 测试创建刷新令牌
        print(f"\n2. 创建刷新令牌")
        refresh_token = JWTUtils.create_refresh_token(user_id)
        print(f"   刷新令牌: {refresh_token[:50]}...")
        
        # 测试验证访问令牌
        print(f"\n3. 验证访问令牌")
        payload = JWTUtils.verify_token(access_token, token_type="access")
        print(f"   用户 ID: {payload['sub']}")
        print(f"   用户名: {payload['username']}")
        print(f"   邮箱: {payload['email']}")
        print(f"   令牌类型: {payload['type']}")
        assert payload['sub'] == str(user_id), "用户 ID 不匹配"
        assert payload['username'] == username, "用户名不匹配"
        
        # 测试验证刷新令牌
        print(f"\n4. 验证刷新令牌")
        payload = JWTUtils.verify_token(refresh_token, token_type="refresh")
        print(f"   用户 ID: {payload['sub']}")
        print(f"   令牌类型: {payload['type']}")
        assert payload['sub'] == str(user_id), "用户 ID 不匹配"
        
        # 测试解码令牌
        print(f"\n5. 解码令牌（不验证）")
        payload = JWTUtils.decode_token(access_token)
        print(f"   用户名: {payload['username']}")
        
        # 测试无效令牌
        print(f"\n6. 测试无效令牌")
        try:
            JWTUtils.verify_token("invalid_token", token_type="access")
            print("   ❌ 应该抛出异常")
            return False
        except Exception as e:
            print(f"   ✅ 正确抛出异常: {str(e)[:50]}...")
        
        print("\n✅ JWT 工具测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ JWT 工具测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*60)
    print("认证系统快速测试")
    print("="*60)
    
    results = []
    
    # 测试密码服务
    results.append(("密码服务", test_password_service()))
    
    # 测试 JWT 工具
    results.append(("JWT 工具", test_jwt_utils()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n下一步:")
        print("1. 运行完整测试: pytest tests/test_auth.py -v")
        print("2. 启动服务: python backend/main.py")
        print("3. 测试 API: 访问 http://localhost:8000/docs")
    else:
        print("\n" + "="*60)
        print("❌ 部分测试失败")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
