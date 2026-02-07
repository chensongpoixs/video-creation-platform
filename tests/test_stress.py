"""
压力测试
"""
import pytest
import time
import threading
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from utils.performance import PerformanceMonitor
from utils.cache import CacheManager


class TestConcurrency:
    """并发测试"""
    
    def test_concurrent_cache_access(self):
        """测试并发缓存访问"""
        cache = CacheManager()
        errors = []
        
        def cache_operation(i):
            try:
                # 写入
                cache.set(f'key{i}', f'value{i}')
                # 读取
                value = cache.get(f'key{i}')
                assert value == f'value{i}'
                # 删除
                cache.delete(f'key{i}')
            except Exception as e:
                errors.append(str(e))
        
        # 并发执行
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(100)]
            for future in as_completed(futures):
                future.result()
        
        assert len(errors) == 0
    
    def test_concurrent_function_calls(self):
        """测试并发函数调用"""
        @PerformanceMonitor.monitor_function
        def test_func(n):
            time.sleep(0.01)
            return n * 2
        
        PerformanceMonitor.reset_metrics()
        
        # 并发调用
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(test_func, i) for i in range(50)]
            results = [future.result() for future in as_completed(futures)]
        
        assert len(results) == 50
        
        metrics = PerformanceMonitor.get_metrics('test_func')
        assert metrics['count'] == 50
        assert metrics['error_count'] == 0


class TestLoadTest:
    """负载测试"""
    
    def test_sustained_load(self):
        """测试持续负载"""
        @PerformanceMonitor.monitor_function
        def work_func(n):
            # 模拟工作负载
            result = sum(range(n))
            return result
        
        PerformanceMonitor.reset_metrics()
        
        # 持续负载（10秒）
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < 10:
            work_func(1000)
            count += 1
        
        metrics = PerformanceMonitor.get_metrics('work_func')
        
        # 验证性能
        assert metrics['count'] == count
        assert metrics['avg_time'] < 0.1  # 平均耗时应该很短
        
        print(f"\n持续负载测试:")
        print(f"  执行次数: {count}")
        print(f"  平均耗时: {metrics['avg_time']:.4f}s")
        print(f"  吞吐量: {count / 10:.1f} ops/s")
    
    def test_burst_load(self):
        """测试突发负载"""
        @PerformanceMonitor.monitor_function
        def burst_func(n):
            time.sleep(0.01)
            return n
        
        PerformanceMonitor.reset_metrics()
        
        # 突发负载
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(burst_func, i) for i in range(100)]
            results = [future.result() for future in as_completed(futures)]
        
        assert len(results) == 100
        
        metrics = PerformanceMonitor.get_metrics('burst_func')
        print(f"\n突发负载测试:")
        print(f"  执行次数: {metrics['count']}")
        print(f"  平均耗时: {metrics['avg_time']:.4f}s")
        print(f"  最大耗时: {metrics['max_time']:.4f}s")


class TestMemoryStress:
    """内存压力测试"""
    
    def test_memory_allocation(self):
        """测试内存分配"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 分配大量内存
        data = []
        for i in range(100):
            data.append([0] * 100000)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 释放内存
        data.clear()
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n内存压力测试:")
        print(f"  初始内存: {initial_memory:.2f} MB")
        print(f"  峰值内存: {peak_memory:.2f} MB")
        print(f"  最终内存: {final_memory:.2f} MB")
        print(f"  内存增长: {peak_memory - initial_memory:.2f} MB")
        
        # 验证内存释放
        assert final_memory < peak_memory


class TestStabilityTest:
    """稳定性测试"""
    
    @pytest.mark.slow
    def test_long_running(self):
        """测试长时间运行（标记为慢速测试）"""
        @PerformanceMonitor.monitor_function
        def work_func():
            time.sleep(0.1)
            return True
        
        PerformanceMonitor.reset_metrics()
        
        # 运行 1 分钟
        start_time = time.time()
        count = 0
        errors = 0
        
        while time.time() - start_time < 60:
            try:
                work_func()
                count += 1
            except Exception:
                errors += 1
        
        metrics = PerformanceMonitor.get_metrics('work_func')
        
        print(f"\n长时间运行测试:")
        print(f"  运行时间: 60秒")
        print(f"  执行次数: {count}")
        print(f"  错误次数: {errors}")
        print(f"  成功率: {metrics['success_count'] / metrics['count'] * 100:.2f}%")
        
        assert errors == 0
        assert metrics['success_count'] == count


class TestResourceLimits:
    """资源限制测试"""
    
    def test_thread_pool_limit(self):
        """测试线程池限制"""
        from utils.async_utils import AsyncTaskManager
        
        manager = AsyncTaskManager(max_workers=5)
        
        def slow_task(n):
            time.sleep(0.1)
            return n
        
        start_time = time.time()
        
        # 提交 20 个任务（超过线程池大小）
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(slow_task, i) for i in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        duration = time.time() - start_time
        
        assert len(results) == 20
        # 应该分批执行，总时间约为 0.4秒（20/5 * 0.1）
        assert duration >= 0.3
        
        print(f"\n线程池限制测试:")
        print(f"  任务数: 20")
        print(f"  线程数: 5")
        print(f"  总耗时: {duration:.2f}s")


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "-s"])
    
    # 只运行快速测试（排除慢速测试）
    # pytest.main([__file__, "-v", "-s", "-m", "not slow"])
