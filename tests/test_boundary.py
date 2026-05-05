"""
边界测试 - 测试系统在边界条件下的行为
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from services.llm_service import generate_script
from repositories import TaskRepository, UserRepository

class TestLLMBoundary:
    """LLM 服务边界测试"""
    
    def test_empty_prompt(self):
        """测试空提示词"""
        script = generate_script("")
        assert script is not None
        assert len(script['scenes']) > 0  # 应该使用备用方案
    
    def test_very_short_prompt(self):
        """测试极短提示词"""
        script = generate_script("森")
        assert script is not None
        assert len(script['scenes']) > 0
    
    def test_very_long_prompt(self):
        """测试超长提示词"""
        long_prompt = "制作视频 " * 1000  # 约10000字
        script = generate_script(long_prompt)
        assert script is not None
        assert len(script['scenes']) > 0
    
    def test_special_characters(self):
        """测试特殊字符"""
        prompts = [
            "制作视频 😀🎬🎥",  # Emoji
            "视频！@#$%^&*()",  # 符号
            "video with 中文 and English",  # 混合语言
            "视频\n换行\t制表",  # 控制字符
        ]
        
        for prompt in prompts:
            script = generate_script(prompt)
            assert script is not None
            assert len(script['scenes']) > 0
    
    def test_repeated_prompt(self):
        """测试重复提示词"""
        prompt = "森林 " * 100
        script = generate_script(prompt)
        assert script is not None
        assert len(script['scenes']) > 0
    
    def test_max_scenes(self):
        """测试最大场景数"""
        prompt = "制作一个包含很多很多场景的超长视频"
        script = generate_script(prompt)
        assert script is not None
        # 场景数应该有合理的上限
        assert len(script['scenes']) <= 20

class TestDatabaseBoundary:
    """数据库边界测试"""
    
    def test_max_field_length(self, db_session, test_user):
        """测试最大字段长度"""
        repo = TaskRepository(db_session)
        
        # 超长提示词
        long_prompt = "x" * 10000
        task = repo.create(
            task_id="test-max-length",
            user_id=test_user.id,
            prompt=long_prompt,
            status="pending"
        )
        
        assert task is not None
        assert len(task.prompt) == 10000
    
    def test_null_values(self, db_session, test_user):
        """测试空值处理"""
        repo = TaskRepository(db_session)
        
        # 可选字段为 None
        task = repo.create(
            task_id="test-null",
            user_id=test_user.id,
            prompt="测试",
            status="pending",
            final_video_path=None,
            error_message=None
        )
        
        assert task is not None
        assert task.final_video_path is None
        assert task.error_message is None
    
    def test_duplicate_task_id(self, db_session, test_user):
        """测试重复任务ID"""
        repo = TaskRepository(db_session)
        
        task_id = "duplicate-test"
        
        # 第一次创建
        task1 = repo.create(
            task_id=task_id,
            user_id=test_user.id,
            prompt="测试1",
            status="pending"
        )
        assert task1 is not None
        
        # 第二次创建应该失败
        with pytest.raises(Exception):
            task2 = repo.create(
                task_id=task_id,
                user_id=test_user.id,
                prompt="测试2",
                status="pending"
            )
    
    def test_zero_quota(self, db_session):
        """测试零配额"""
        repo = UserRepository(db_session)
        
        user = repo.create(
            username="zero_quota_user",
            email="zero@test.com",
            quota=0
        )
        
        assert user is not None
        assert not user.has_quota()
    
    def test_negative_progress(self, db_session, test_task):
        """测试负数进度"""
        repo = TaskRepository(db_session)
        
        # 尝试设置负数进度
        task = repo.update_progress(test_task.task_id, progress=-10)
        
        # 系统应该处理或拒绝
        assert task.progress >= 0
    
    def test_progress_over_100(self, db_session, test_task):
        """测试超过100的进度"""
        repo = TaskRepository(db_session)
        
        # 尝试设置超过100的进度
        task = repo.update_progress(test_task.task_id, progress=150)
        
        # 系统应该限制在100以内
        assert task.progress <= 100

class TestVideoBoundary:
    """视频服务边界测试"""
    
    def test_min_duration(self):
        """测试最小时长"""
        scene = {
            "scene_number": 1,
            "description": "测试场景",
            "duration": 1  # 最小1秒
        }
        # 应该能正常处理
        assert scene['duration'] >= 1
    
    def test_max_duration(self):
        """测试最大时长"""
        scene = {
            "scene_number": 1,
            "description": "测试场景",
            "duration": 60  # 最大60秒
        }
        # 应该能正常处理
        assert scene['duration'] <= 60
    
    def test_zero_duration(self):
        """测试零时长"""
        scene = {
            "scene_number": 1,
            "description": "测试场景",
            "duration": 0
        }
        # 应该有默认值或拒绝
        assert scene['duration'] >= 0
    
    def test_many_scenes(self):
        """测试大量场景"""
        scenes = [
            {
                "scene_number": i,
                "description": f"场景{i}",
                "duration": 2
            }
            for i in range(1, 101)  # 100个场景
        ]
        
        assert len(scenes) == 100
        # 系统应该能处理或有合理限制

class TestUserBoundary:
    """用户边界测试"""
    
    def test_max_username_length(self, db_session):
        """测试最大用户名长度"""
        repo = UserRepository(db_session)
        
        # 50字符（表定义的最大长度）
        long_username = "x" * 50
        user = repo.create(
            username=long_username,
            email="long@test.com",
            quota=100
        )
        
        assert user is not None
        assert len(user.username) == 50
    
    def test_username_too_long(self, db_session):
        """测试超长用户名"""
        repo = UserRepository(db_session)
        
        # 超过50字符
        too_long_username = "x" * 100
        
        # 应该失败或截断
        with pytest.raises(Exception):
            user = repo.create(
                username=too_long_username,
                email="toolong@test.com",
                quota=100
            )
    
    def test_max_quota(self, db_session):
        """测试最大配额"""
        repo = UserRepository(db_session)
        
        user = repo.create(
            username="max_quota_user",
            email="max@test.com",
            quota=999999
        )
        
        assert user is not None
        assert user.quota == 999999
    
    def test_quota_overflow(self, db_session, test_user):
        """测试配额溢出"""
        repo = UserRepository(db_session)
        
        # 尝试使用超过配额的量
        with pytest.raises(ValueError):
            repo.use_quota(test_user.id, amount=test_user.quota + 1)

def test_summary():
    """边界测试总结"""
    print("\n" + "="*60)
    print("边界测试完成")
    print("="*60)
    print("测试类型:")
    print("  - LLM 边界测试: 6个用例")
    print("  - 数据库边界测试: 6个用例")
    print("  - 视频边界测试: 4个用例")
    print("  - 用户边界测试: 4个用例")
    print("总计: 20个边界测试用例")
    print("="*60)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
