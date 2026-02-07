"""
集成测试 - 测试模块间的协作
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
import uuid
from models import get_db_context, Script, Video
from repositories import TaskRepository, UserRepository, VideoRepository
from services.llm_service import generate_script

class TestCompleteWorkflow:
    """完整业务流程测试"""
    
    def test_user_to_video_workflow(self, db_session):
        """测试：用户注册 → 创建任务 → 生成视频 → 查询结果"""
        
        # 1. 创建用户
        user_repo = UserRepository(db_session)
        user = user_repo.create(
            username=f"workflow_user_{uuid.uuid4().hex[:8]}",
            email=f"workflow_{uuid.uuid4().hex[:8]}@test.com",
            quota=10
        )
        assert user is not None
        print(f"\n✅ 步骤1: 用户创建成功 (ID: {user.id})")
        
        # 2. 检查配额
        assert user.has_quota()
        print(f"✅ 步骤2: 配额检查通过 ({user.quota})")
        
        # 3. 创建任务
        task_repo = TaskRepository(db_session)
        task = task_repo.create(
            task_id=str(uuid.uuid4()),
            user_id=user.id,
            prompt="制作测试视频",
            status="pending",
            total_scenes=3
        )
        assert task is not None
        print(f"✅ 步骤3: 任务创建成功 (ID: {task.task_id})")
        
        # 4. 使用配额
        user_repo.use_quota(user.id, 1)
        user = user_repo.get(user.id)
        assert user.used_quota == 1
        print(f"✅ 步骤4: 配额使用成功 (剩余: {user.quota - user.used_quota})")
        
        # 5. 开始任务
        task_repo.start_task(task.task_id)
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "processing"
        print(f"✅ 步骤5: 任务已开始")
        
        # 6. 生成脚本
        for i in range(3):
            script = Script(
                task_id=task.id,
                scene_number=i+1,
                description=f"场景{i+1}",
                duration=3
            )
            db_session.add(script)
        db_session.commit()
        print(f"✅ 步骤6: 脚本生成完成 (3个场景)")
        
        # 7. 生成视频
        video_repo = VideoRepository(db_session)
        for i in range(3):
            video = video_repo.create(
                task_id=task.id,
                scene_number=i+1,
                file_path=f"/test/scene_{i+1}.mp4",
                file_size=1024*1024,
                duration=3.0
            )
        print(f"✅ 步骤7: 视频生成完成 (3个视频)")
        
        # 8. 完成任务
        task_repo.complete_task(task.task_id, "/test/final.mp4")
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "completed"
        assert task.final_video_path == "/test/final.mp4"
        print(f"✅ 步骤8: 任务完成")
        
        # 9. 查询结果
        task_with_data = task_repo.get_with_relations(task.task_id)
        assert len(task_with_data.scripts) == 3
        assert len(task_with_data.videos) == 3
        print(f"✅ 步骤9: 结果查询成功")
        
        print(f"\n✅ 完整流程测试通过！")
    
    def test_quota_limit_workflow(self, db_session):
        """测试：配额管理 → 任务限制 → 配额恢复"""
        
        # 1. 创建配额有限的用户
        user_repo = UserRepository(db_session)
        user = user_repo.create(
            username=f"quota_user_{uuid.uuid4().hex[:8]}",
            email=f"quota_{uuid.uuid4().hex[:8]}@test.com",
            quota=2  # 只有2次配额
        )
        print(f"\n✅ 用户创建: 配额={user.quota}")
        
        # 2. 使用配额创建任务
        task_repo = TaskRepository(db_session)
        
        for i in range(2):
            task = task_repo.create(
                task_id=str(uuid.uuid4()),
                user_id=user.id,
                prompt=f"任务{i+1}",
                status="pending"
            )
            user_repo.use_quota(user.id, 1)
            print(f"✅ 任务{i+1}创建，配额使用")
        
        # 3. 验证配额用尽
        user = user_repo.get(user.id)
        assert not user.has_quota()
        print(f"✅ 配额已用尽: {user.used_quota}/{user.quota}")
        
        # 4. 尝试创建新任务（应该失败）
        with pytest.raises(ValueError):
            user_repo.use_quota(user.id, 1)
        print(f"✅ 配额限制生效")
        
        # 5. 重置配额
        user_repo.reset_quota(user.id, quota=10)
        user = user_repo.get(user.id)
        assert user.has_quota()
        assert user.used_quota == 0
        print(f"✅ 配额重置: {user.quota}")
    
    def test_error_recovery_workflow(self, db_session, test_user):
        """测试：错误处理 → 重试 → 成功"""
        
        task_repo = TaskRepository(db_session)
        
        # 1. 创建任务
        task = task_repo.create(
            task_id=str(uuid.uuid4()),
            user_id=test_user.id,
            prompt="错误恢复测试",
            status="pending",
            total_scenes=3
        )
        print(f"\n✅ 任务创建")
        
        # 2. 开始任务
        task_repo.start_task(task.task_id)
        print(f"✅ 任务开始")
        
        # 3. 模拟失败
        task_repo.fail_task(task.task_id, "模拟错误")
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "failed"
        print(f"✅ 任务失败: {task.error_message}")
        
        # 4. 重试：重置状态
        task_repo.update_status(task.task_id, "pending")
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "pending"
        print(f"✅ 任务重置为待处理")
        
        # 5. 再次开始
        task_repo.start_task(task.task_id)
        print(f"✅ 任务重新开始")
        
        # 6. 成功完成
        task_repo.complete_task(task.task_id, "/test/success.mp4")
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "completed"
        print(f"✅ 任务成功完成")

class TestDataFlow:
    """数据流测试"""
    
    def test_prompt_to_video_flow(self, db_session, test_user):
        """测试：提示词 → 脚本 → 视频 → 存储"""
        
        # 1. 输入提示词
        prompt = "制作一段森林探险视频"
        print(f"\n✅ 输入提示词: {prompt}")
        
        # 2. 生成脚本
        script_data = generate_script(prompt)
        assert 'scenes' in script_data
        print(f"✅ 脚本生成: {len(script_data['scenes'])}个场景")
        
        # 3. 创建任务
        task_repo = TaskRepository(db_session)
        task = task_repo.create(
            task_id=str(uuid.uuid4()),
            user_id=test_user.id,
            prompt=prompt,
            status="processing",
            total_scenes=len(script_data['scenes'])
        )
        print(f"✅ 任务创建: {task.task_id}")
        
        # 4. 保存脚本
        for scene in script_data['scenes']:
            script = Script(
                task_id=task.id,
                scene_number=scene['scene_number'],
                description=scene['description'],
                duration=scene['duration']
            )
            db_session.add(script)
        db_session.commit()
        print(f"✅ 脚本保存到数据库")
        
        # 5. 生成视频（模拟）
        video_repo = VideoRepository(db_session)
        for scene in script_data['scenes']:
            video = video_repo.create(
                task_id=task.id,
                scene_number=scene['scene_number'],
                file_path=f"/test/scene_{scene['scene_number']}.mp4",
                file_size=1024*1024*5,
                duration=float(scene['duration'])
            )
        print(f"✅ 视频生成并保存")
        
        # 6. 验证数据完整性
        task_with_data = task_repo.get_with_relations(task.task_id)
        assert len(task_with_data.scripts) == len(script_data['scenes'])
        assert len(task_with_data.videos) == len(script_data['scenes'])
        print(f"✅ 数据完整性验证通过")
    
    def test_status_update_flow(self, db_session, test_user):
        """测试：任务状态 → 进度更新 → 完成通知"""
        
        task_repo = TaskRepository(db_session)
        
        # 1. 创建任务
        task = task_repo.create(
            task_id=str(uuid.uuid4()),
            user_id=test_user.id,
            prompt="状态流测试",
            status="pending",
            total_scenes=5
        )
        print(f"\n✅ 任务创建: pending")
        
        # 2. 开始处理
        task_repo.start_task(task.task_id)
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "processing"
        assert task.started_at is not None
        print(f"✅ 任务开始: processing")
        
        # 3. 进度更新
        for i in range(1, 6):
            progress = int((i / 5) * 100)
            task_repo.update_progress(task.task_id, progress=progress, completed_scenes=i)
            task = task_repo.get_by_task_id(task.task_id)
            print(f"✅ 进度更新: {task.progress}% ({task.completed_scenes}/{task.total_scenes})")
        
        # 4. 完成任务
        task_repo.complete_task(task.task_id, "/test/final.mp4")
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "completed"
        assert task.completed_at is not None
        assert task.duration is not None
        print(f"✅ 任务完成: completed (耗时: {task.duration:.2f}秒)")

class TestModuleIntegration:
    """模块集成测试"""
    
    def test_llm_database_integration(self, db_session, test_user):
        """测试 LLM 服务与数据库的集成"""
        
        # 1. LLM 生成脚本
        prompt = "制作视频"
        script_data = generate_script(prompt)
        print(f"\n✅ LLM 生成脚本: {len(script_data['scenes'])}个场景")
        
        # 2. 保存到数据库
        task_repo = TaskRepository(db_session)
        task = task_repo.create(
            task_id=str(uuid.uuid4()),
            user_id=test_user.id,
            prompt=prompt,
            status="processing",
            total_scenes=len(script_data['scenes'])
        )
        
        for scene in script_data['scenes']:
            script = Script(
                task_id=task.id,
                scene_number=scene['scene_number'],
                description=scene['description'],
                duration=scene['duration']
            )
            db_session.add(script)
        db_session.commit()
        print(f"✅ 脚本保存到数据库")
        
        # 3. 从数据库读取
        task_with_scripts = task_repo.get_with_relations(task.task_id)
        assert len(task_with_scripts.scripts) == len(script_data['scenes'])
        print(f"✅ 从数据库读取脚本")
        
        # 4. 验证数据一致性
        for i, script in enumerate(task_with_scripts.scripts):
            original = script_data['scenes'][i]
            assert script.scene_number == original['scene_number']
            assert script.description == original['description']
            assert script.duration == original['duration']
        print(f"✅ 数据一致性验证通过")
    
    def test_repository_service_integration(self, db_session, test_user):
        """测试仓储层与服务层的集成"""
        
        # 使用仓储层
        task_repo = TaskRepository(db_session)
        user_repo = UserRepository(db_session)
        
        # 1. 检查用户配额
        user = user_repo.get(test_user.id)
        initial_quota = user.quota - user.used_quota
        print(f"\n✅ 初始配额: {initial_quota}")
        
        # 2. 创建任务
        task = task_repo.create(
            task_id=str(uuid.uuid4()),
            user_id=test_user.id,
            prompt="集成测试",
            status="pending"
        )
        print(f"✅ 任务创建")
        
        # 3. 使用配额
        user_repo.use_quota(test_user.id, 1)
        user = user_repo.get(test_user.id)
        assert user.used_quota == test_user.used_quota + 1
        print(f"✅ 配额使用")
        
        # 4. 完成任务
        task_repo.complete_task(task.task_id)
        task = task_repo.get_by_task_id(task.task_id)
        assert task.status == "completed"
        print(f"✅ 任务完成")

def test_summary():
    """集成测试总结"""
    print("\n" + "="*60)
    print("集成测试完成")
    print("="*60)
    print("测试类型:")
    print("  - 完整业务流程: 3个用例")
    print("  - 数据流测试: 2个用例")
    print("  - 模块集成: 2个用例")
    print("总计: 7个集成测试用例")
    print("="*60)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
