"""
数据库测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import uuid
from datetime import datetime
from models import get_db_context, User, Task, Script, Video, TaskStatus
from repositories import TaskRepository, UserRepository, VideoRepository

def test_user_operations():
    """测试用户操作"""
    print("\n" + "="*60)
    print("测试用户操作")
    print("="*60)
    
    with get_db_context() as db:
        user_repo = UserRepository(db)
        
        # 创建用户
        print("\n1. 创建用户...")
        user = user_repo.create(
            username=f"test_user_{uuid.uuid4().hex[:8]}",
            email=f"test_{uuid.uuid4().hex[:8]}@example.com",
            quota=100
        )
        print(f"✅ 用户创建成功: {user.username}")
        
        # 查询用户
        print("\n2. 查询用户...")
        found_user = user_repo.get(user.id)
        print(f"✅ 查询成功: {found_user.username}")
        
        # 使用配额
        print("\n3. 使用配额...")
        user_repo.use_quota(user.id, 5)
        updated_user = user_repo.get(user.id)
        print(f"✅ 配额使用成功: {updated_user.used_quota}/{ updated_user.quota}")
        
        return user.id

def test_task_operations(user_id):
    """测试任务操作"""
    print("\n" + "="*60)
    print("测试任务操作")
    print("="*60)
    
    with get_db_context() as db:
        task_repo = TaskRepository(db)
        
        # 创建任务
        print("\n1. 创建任务...")
        task_id = str(uuid.uuid4())
        task = task_repo.create(
            task_id=task_id,
            user_id=user_id,
            prompt="测试视频生成",
            status=TaskStatus.PENDING.value,
            total_scenes=3
        )
        print(f"✅ 任务创建成功: {task.task_id}")
        
        # 开始任务
        print("\n2. 开始任务...")
        task_repo.start_task(task_id)
        task = task_repo.get_by_task_id(task_id)
        print(f"✅ 任务已开始: {task.status}")
        
        # 更新进度
        print("\n3. 更新进度...")
        task_repo.update_progress(task_id, progress=50, completed_scenes=1)
        task = task_repo.get_by_task_id(task_id)
        print(f"✅ 进度更新: {task.progress}% ({task.completed_scenes}/{task.total_scenes})")
        
        # 完成任务
        print("\n4. 完成任务...")
        task_repo.complete_task(task_id, "/path/to/video.mp4")
        task = task_repo.get_by_task_id(task_id)
        print(f"✅ 任务完成: {task.status}, 耗时: {task.duration}秒")
        
        return task.id

def test_script_operations(task_id):
    """测试脚本操作"""
    print("\n" + "="*60)
    print("测试脚本操作")
    print("="*60)
    
    with get_db_context() as db:
        # 创建脚本
        print("\n1. 创建脚本...")
        scripts = []
        for i in range(3):
            script = Script(
                task_id=task_id,
                scene_number=i+1,
                description=f"场景 {i+1} 描述",
                duration=3
            )
            db.add(script)
            scripts.append(script)
        
        db.commit()
        print(f"✅ 创建了 {len(scripts)} 个脚本")
        
        # 查询脚本
        print("\n2. 查询脚本...")
        found_scripts = db.query(Script).filter(Script.task_id == task_id).all()
        print(f"✅ 查询到 {len(found_scripts)} 个脚本")
        for script in found_scripts:
            print(f"   场景 {script.scene_number}: {script.description}")

def test_video_operations(task_id):
    """测试视频操作"""
    print("\n" + "="*60)
    print("测试视频操作")
    print("="*60)
    
    with get_db_context() as db:
        video_repo = VideoRepository(db)
        
        # 创建视频
        print("\n1. 创建视频...")
        videos = []
        for i in range(3):
            video = video_repo.create(
                task_id=task_id,
                scene_number=i+1,
                file_path=f"/path/to/scene_{i+1}.mp4",
                file_size=1024*1024*5,  # 5MB
                duration=3.0,
                width=1024,
                height=576,
                fps=6
            )
            videos.append(video)
        
        print(f"✅ 创建了 {len(videos)} 个视频")
        
        # 查询视频
        print("\n2. 查询视频...")
        found_videos = video_repo.get_by_task(task_id)
        print(f"✅ 查询到 {len(found_videos)} 个视频")
        for video in found_videos:
            print(f"   场景 {video.scene_number}: {video.file_path} ({video.get_file_size_mb()} MB)")

def test_query_operations(user_id):
    """测试查询操作"""
    print("\n" + "="*60)
    print("测试查询操作")
    print("="*60)
    
    with get_db_context() as db:
        task_repo = TaskRepository(db)
        
        # 查询用户任务
        print("\n1. 查询用户任务...")
        tasks = task_repo.get_by_user(user_id, limit=10)
        print(f"✅ 查询到 {len(tasks)} 个任务")
        
        # 统计任务
        print("\n2. 统计任务...")
        stats = task_repo.get_statistics()
        print(f"✅ 统计结果:")
        print(f"   总任务数: {stats['total']}")
        print(f"   已完成: {stats['completed']}")
        print(f"   失败: {stats['failed']}")
        print(f"   成功率: {stats['success_rate']}%")
        print(f"   平均耗时: {stats['avg_duration']}秒")

def test_relations():
    """测试关联查询"""
    print("\n" + "="*60)
    print("测试关联查询")
    print("="*60)
    
    with get_db_context() as db:
        task_repo = TaskRepository(db)
        
        # 获取最近的任务
        print("\n1. 获取最近任务...")
        tasks = task_repo.get_recent(limit=1)
        
        if tasks:
            task = tasks[0]
            print(f"✅ 任务: {task.task_id}")
            
            # 获取关联数据
            print("\n2. 获取关联数据...")
            task_with_relations = task_repo.get_with_relations(task.task_id)
            
            if task_with_relations:
                print(f"✅ 脚本数: {len(task_with_relations.scripts)}")
                print(f"✅ 视频数: {len(task_with_relations.videos)}")
                
                # 转换为字典
                print("\n3. 转换为字典...")
                task_dict = task_with_relations.to_dict(include_relations=True)
                print(f"✅ 字典包含 {len(task_dict)} 个字段")

def main():
    """主函数"""
    print("="*60)
    print("数据库功能测试")
    print("="*60)
    
    try:
        # 测试用户操作
        user_id = test_user_operations()
        
        # 测试任务操作
        task_id = test_task_operations(user_id)
        
        # 测试脚本操作
        test_script_operations(task_id)
        
        # 测试视频操作
        test_video_operations(task_id)
        
        # 测试查询操作
        test_query_operations(user_id)
        
        # 测试关联查询
        test_relations()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
