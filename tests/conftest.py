"""
Pytest 配置和通用 Fixture
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
import uuid
from datetime import datetime
from models import get_db_context, init_db, User, Task, Script, Video, TaskStatus
from repositories import UserRepository, TaskRepository, VideoRepository

# 测试数据库初始化（会话级别，只执行一次）
@pytest.fixture(scope="session", autouse=True)
def init_test_db():
    """初始化测试数据库"""
    try:
        init_db()
        print("\n✅ 测试数据库初始化完成")
    except Exception as e:
        print(f"\n⚠️ 数据库已存在或初始化失败: {e}")
    yield

# 数据库会话 Fixture
@pytest.fixture
def db_session():
    """提供数据库会话"""
    with get_db_context() as db:
        yield db

# 测试用户 Fixture
@pytest.fixture
def test_user(db_session):
    """创建测试用户"""
    repo = UserRepository(db_session)
    user = repo.create(
        username=f"test_user_{uuid.uuid4().hex[:8]}",
        email=f"test_{uuid.uuid4().hex[:8]}@example.com",
        quota=100,
        is_active=True
    )
    yield user
    # 清理（可选）

# 测试任务 Fixture
@pytest.fixture
def test_task(db_session, test_user):
    """创建测试任务"""
    repo = TaskRepository(db_session)
    task = repo.create(
        task_id=str(uuid.uuid4()),
        user_id=test_user.id,
        prompt="测试提示词",
        status=TaskStatus.PENDING.value,
        total_scenes=3
    )
    yield task

# 完整任务（带脚本和视频）Fixture
@pytest.fixture
def complete_task(db_session, test_task):
    """创建完整的任务（带脚本和视频）"""
    # 添加脚本
    for i in range(3):
        script = Script(
            task_id=test_task.id,
            scene_number=i+1,
            description=f"测试场景 {i+1}",
            duration=3
        )
        db_session.add(script)
    
    # 添加视频
    video_repo = VideoRepository(db_session)
    for i in range(3):
        video_repo.create(
            task_id=test_task.id,
            scene_number=i+1,
            file_path=f"/test/scene_{i+1}.mp4",
            file_size=1024*1024,
            duration=3.0
        )
    
    db_session.commit()
    yield test_task

# Mock LLM 响应
@pytest.fixture
def mock_llm_response():
    """Mock LLM 响应"""
    return {
        "title": "测试视频",
        "total_duration": 9,
        "scenes": [
            {
                "scene_number": 1,
                "description": "场景1描述",
                "duration": 3,
                "camera_movement": "静止",
                "lighting": "自然光"
            },
            {
                "scene_number": 2,
                "description": "场景2描述",
                "duration": 3,
                "camera_movement": "推进",
                "lighting": "柔光"
            },
            {
                "scene_number": 3,
                "description": "场景3描述",
                "duration": 3,
                "camera_movement": "环绕",
                "lighting": "背光"
            }
        ]
    }

# 测试数据生成器
class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_prompt(length=50):
        """生成测试提示词"""
        return "测试提示词 " * (length // 7)
    
    @staticmethod
    def generate_user_data():
        """生成用户数据"""
        uid = uuid.uuid4().hex[:8]
        return {
            "username": f"user_{uid}",
            "email": f"user_{uid}@test.com",
            "quota": 100
        }
    
    @staticmethod
    def generate_task_data(user_id=1):
        """生成任务数据"""
        return {
            "task_id": str(uuid.uuid4()),
            "user_id": user_id,
            "prompt": "测试提示词",
            "status": TaskStatus.PENDING.value,
            "total_scenes": 3
        }

@pytest.fixture
def data_generator():
    """提供数据生成器"""
    return TestDataGenerator()

# 性能计时器
@pytest.fixture
def timer():
    """性能计时器"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
    
    return Timer()

# 临时文件清理
@pytest.fixture
def temp_files():
    """临时文件管理"""
    files = []
    
    def add_file(filepath):
        files.append(filepath)
        return filepath
    
    yield add_file
    
    # 清理
    import os
    for filepath in files:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
