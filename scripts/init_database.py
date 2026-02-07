"""
数据库初始化脚本
"""
import sys
import os

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models import init_db, get_db_info, User, get_db_context
from utils.logger import setup_logger
import secrets

logger = setup_logger(__name__)

def create_default_user():
    """创建默认用户"""
    try:
        with get_db_context() as db:
            # 检查是否已有用户
            from repositories.user_repository import UserRepository
            user_repo = UserRepository(db)
            
            if user_repo.username_exists("admin"):
                logger.info("默认用户已存在，跳过创建")
                return
            
            # 创建默认用户
            api_key = secrets.token_urlsafe(32)
            user = user_repo.create(
                username="admin",
                email="admin@example.com",
                api_key=api_key,
                quota=1000,
                is_active=True
            )
            
            logger.info(f"✅ 创建默认用户成功")
            logger.info(f"   用户名: {user.username}")
            logger.info(f"   邮箱: {user.email}")
            logger.info(f"   API Key: {api_key}")
            logger.info(f"   配额: {user.quota}")
            
    except Exception as e:
        logger.error(f"❌ 创建默认用户失败: {str(e)}")
        raise

def main():
    """主函数"""
    print("="*60)
    print("数据库初始化")
    print("="*60)
    
    try:
        # 初始化数据库
        print("\n1. 创建数据库表...")
        init_db()
        
        # 创建默认用户
        print("\n2. 创建默认用户...")
        create_default_user()
        
        # 显示数据库信息
        print("\n3. 数据库信息:")
        info = get_db_info()
        print(f"   URL: {info['url']}")
        print(f"   表: {', '.join(info['tables'])}")
        if 'size_mb' in info:
            print(f"   大小: {info['size_mb']} MB")
        
        print("\n" + "="*60)
        print("✅ 数据库初始化完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
