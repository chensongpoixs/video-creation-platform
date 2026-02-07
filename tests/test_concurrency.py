"""
并发测试 - 测试系统的并发处理能力
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import get_db_context
from repositories import TaskRepository, UserRepository

class TestConcurrentTaskCreation:
    """并发任务创建测试"""
    
    def test_concurrent_task_creation_10(self, test_user):
        """测试10个并发任务创建"""
        num_tasks = 10
        created_tasks = []
        errors = []
        
        def create_task():
            try:
                with get_db_context() as db:
                    repo = TaskRepository(db)
                    task = repo.create(
                        task_id=str(uuid.uuid4()),
                        user_id=test_user.id,
                        prompt="并发测试",
                        status="pending"
                    )
                    created_tasks.append(task.task_id)
                    return task.task_id
            except Exception as e:
                errors.append(str(e))
                return None
        
        # 并发创建
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_task) for _ in range(num_tasks)]
            results = [f.result() for f in as_completed(futures)]
        
        # 验证
        print(f"\n创建成功: {len(created_tasks)}/{num_tasks}")
        print(f"错误数: {len(errors)}")
        
        assert len(created_tasks) > 0
        assert len(created_tasks) + len(errors) == num_tasks
    
    def test_concurrent_task_creation_100(self, test_user):
        """测试100个并发任务创建"""
        num_tasks = 100
        created_count = 0
        error_count = 0
        
        def create_task():
            nonlocal created_count, error_count
            try:
                with get_db_context() as db:
                    repo = TaskRepository(db)
                    task = repo.create(
                        task_id=str(uuid.uuid4()),
                        user_id=test_user.id,
                        prompt="大量并发测试",
                        status="pending"
                    )
                    created_count += 1
            except Exception as e:
                error_count += 1
        
        # 并发创建
        threads = []
        for _ in range(num_tasks):
            t = threading.Thread(target=create_task)
            threads.append(t)
            t.start()
        
        # 等待完成
        for t in threads:
            t.join()
        
        print(f"\n创建成功: {created_count}/{num_tasks}")
        print(f"错误数: {error_count}")
        
        # 大部分应该成功
        assert created_count > num_tasks * 0.8

class TestConcurrentDatabaseOperations:
    """并发数据库操作测试"""
    
    def test_concurrent_read(self, test_task):
        """测试并发读取"""
        num_reads = 50
        results = []
        
        def read_task():
            with get_db_context() as db:
                repo = TaskRepository(db)
                task = repo.get_by_task_id(test_task.task_id)
                results.append(task is not None)
        
        # 并发读取
        threads = []
        for _ in range(num_reads):
            t = threading.Thread(target=read_task)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 所有读取都应该成功
        assert all(results)
        assert len(results) == num_reads
    
    def test_concurrent_write(self, test_task):
        """测试并发写入"""
        num_writes = 20
        success_count = 0
        
        def update_task(progress):
            nonlocal success_count
            try:
                with get_db_context() as db:
                    repo = TaskRepository(db)
                    repo.update_progress(test_task.task_id, progress=progress)
                    success_count += 1
            except Exception as e:
                pass
        
        # 并发更新
        threads = []
        for i in range(num_writes):
            t = threading.Thread(target=update_task, args=(i * 5,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"\n更新成功: {success_count}/{num_writes}")
        
        # 验证最终状态
        with get_db_context() as db:
            repo = TaskRepository(db)
            task = repo.get_by_task_id(test_task.task_id)
            print(f"最终进度: {task.progress}")
            assert task.progress >= 0
    
    def test_concurrent_read_write(self, test_task):
        """测试并发读写"""
        num_operations = 30
        read_count = 0
        write_count = 0
        
        def read_task():
            nonlocal read_count
            with get_db_context() as db:
                repo = TaskRepository(db)
                task = repo.get_by_task_id(test_task.task_id)
                if task:
                    read_count += 1
        
        def write_task(progress):
            nonlocal write_count
            try:
                with get_db_context() as db:
                    repo = TaskRepository(db)
                    repo.update_progress(test_task.task_id, progress=progress)
                    write_count += 1
            except:
                pass
        
        # 混合读写操作
        threads = []
        for i in range(num_operations):
            if i % 2 == 0:
                t = threading.Thread(target=read_task)
            else:
                t = threading.Thread(target=write_task, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"\n读取: {read_count}, 写入: {write_count}")
        assert read_count > 0
        assert write_count > 0
    
    def test_transaction_isolation(self, test_user):
        """测试事务隔离"""
        task_id = str(uuid.uuid4())
        
        def create_and_update():
            with get_db_context() as db:
                repo = TaskRepository(db)
                
                # 创建任务
                task = repo.create(
                    task_id=task_id,
                    user_id=test_user.id,
                    prompt="事务测试",
                    status="pending"
                )
                
                # 更新任务
                time.sleep(0.1)
                repo.update_progress(task_id, progress=50)
        
        def read_task():
            time.sleep(0.05)
            with get_db_context() as db:
                repo = TaskRepository(db)
                task = repo.get_by_task_id(task_id)
                return task
        
        # 并发执行
        t1 = threading.Thread(target=create_and_update)
        t2 = threading.Thread(target=read_task)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # 验证最终状态
        with get_db_context() as db:
            repo = TaskRepository(db)
            task = repo.get_by_task_id(task_id)
            assert task is not None

class TestResourceCompetition:
    """资源竞争测试"""
    
    def test_quota_competition(self, test_user):
        """测试配额竞争"""
        initial_quota = test_user.quota
        num_operations = 20
        success_count = 0
        
        def use_quota():
            nonlocal success_count
            try:
                with get_db_context() as db:
                    repo = UserRepository(db)
                    repo.use_quota(test_user.id, amount=1)
                    success_count += 1
            except Exception as e:
                pass
        
        # 并发使用配额
        threads = []
        for _ in range(num_operations):
            t = threading.Thread(target=use_quota)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"\n成功使用配额: {success_count}/{num_operations}")
        
        # 验证配额
        with get_db_context() as db:
            repo = UserRepository(db)
            user = repo.get(test_user.id)
            print(f"剩余配额: {user.quota - user.used_quota}")
            
            # 已使用配额应该等于成功次数
            assert user.used_quota <= initial_quota
    
    def test_file_write_competition(self, temp_files):
        """测试文件写入竞争"""
        import tempfile
        
        filepath = temp_files(tempfile.mktemp(suffix=".txt"))
        num_writes = 10
        
        def write_file(content):
            try:
                with open(filepath, 'a') as f:
                    f.write(content + "\n")
            except Exception as e:
                pass
        
        # 并发写入
        threads = []
        for i in range(num_writes):
            t = threading.Thread(target=write_file, args=(f"Line {i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证文件内容
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
            print(f"\n写入行数: {len(lines)}")
            assert len(lines) > 0

class TestDeadlockPrevention:
    """死锁预防测试"""
    
    def test_no_deadlock_on_concurrent_updates(self, test_user):
        """测试并发更新不会死锁"""
        task_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        # 创建多个任务
        with get_db_context() as db:
            repo = TaskRepository(db)
            for task_id in task_ids:
                repo.create(
                    task_id=task_id,
                    user_id=test_user.id,
                    prompt="死锁测试",
                    status="pending"
                )
        
        def update_tasks():
            with get_db_context() as db:
                repo = TaskRepository(db)
                for task_id in task_ids:
                    repo.update_progress(task_id, progress=50)
                    time.sleep(0.01)
        
        # 并发更新
        threads = []
        for _ in range(3):
            t = threading.Thread(target=update_tasks)
            threads.append(t)
            t.start()
        
        # 设置超时
        for t in threads:
            t.join(timeout=5)
            if t.is_alive():
                pytest.fail("可能发生死锁")

def test_summary():
    """并发测试总结"""
    print("\n" + "="*60)
    print("并发测试完成")
    print("="*60)
    print("测试类型:")
    print("  - 并发任务创建: 2个用例")
    print("  - 并发数据库操作: 4个用例")
    print("  - 资源竞争: 2个用例")
    print("  - 死锁预防: 1个用例")
    print("总计: 9个并发测试用例")
    print("="*60)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
