"""
性能测试
"""
import pytest
import time
import sys
import os

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from utils.performance import PerformanceMonitor, ResourceMonitor
from utils.cache import MemoryCache, CacheManager, cached
from utils.async_utils import AsyncTaskManager, BatchProcessor


class TestPerformanceMonitor:
    """性能监控测试"""
    
    def test_monitor_function(self):
        """测试函数监控"""
        @PerformanceMonitor.monitor_function
        def test_func(n):
            time.sleep(0.1)
            return n * 2
        
        # 重置指标
        PerformanceMonitor.reset_metrics()
        
        # 执行函数
        result = test_func(5)
        assert result == 10
        
        # 检查指标
        metrics = PerformanceMonitor.get_metrics('test_func')
        assert metrics['count'] == 1
        assert metrics['avg_time'] >= 0.1
        assert metrics['success_count'] == 1
    
    def test_monitor_multiple_calls(self):
        """测试多次调用监控"""
        @PerformanceMonitor.monitor_function
        def test_func(n):
            time.sleep(0.01)
            return n
        
        PerformanceMonitor.reset_metrics()
        
        # 多次调用
        for i in range(10):
            test_func(i)
        
        metrics = PerformanceMonitor.get_metrics('test_func')
        assert metrics['count'] == 10
        assert metrics['min_time'] > 0
        assert metrics['max_time'] > metrics['min_time']
    
    def test_monitor_error(self):
        """测试错误监控"""
        @PerformanceMonitor.monitor_function
        def test_func():
            raise ValueError("Test error")
        
        PerformanceMonitor.reset_metrics()
        
        with pytest.raises(ValueError):
            test_func()
        
        metrics = PerformanceMonitor.get_metrics('test_func')
        assert metrics['error_count'] == 1
        assert len(metrics['errors']) > 0
    
    def test_get_system_metrics(self):
        """测试系统指标获取"""
        metrics = PerformanceMonitor.get_system_metrics()
        
        assert 'cpu' in metrics
        assert 'memory' in metrics
        assert 'disk' in metrics
        assert metrics['cpu']['percent'] >= 0
        assert metrics['memory']['percent'] >= 0


class TestMemoryCache:
    """内存缓存测试"""
    
    def test_cache_set_get(self):
        """测试缓存设置和获取"""
        cache = MemoryCache()
        
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
    
    def test_cache_expiration(self):
        """测试缓存过期"""
        cache = MemoryCache()
        
        cache.set('key1', 'value1', ttl=1)
        assert cache.get('key1') == 'value1'
        
        time.sleep(1.1)
        assert cache.get('key1') is None
    
    def test_cache_delete(self):
        """测试缓存删除"""
        cache = MemoryCache()
        
        cache.set('key1', 'value1')
        cache.delete('key1')
        assert cache.get('key1') is None
    
    def test_cache_clear(self):
        """测试缓存清空"""
        cache = MemoryCache()
        
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.clear()
        
        assert cache.get('key1') is None
        assert cache.get('key2') is None
    
    def test_cache_max_size(self):
        """测试缓存大小限制"""
        cache = MemoryCache(max_size=10)
        
        # 添加超过最大数量的缓存
        for i in range(15):
            cache.set(f'key{i}', f'value{i}')
        
        stats = cache.get_stats()
        assert stats['size'] <= 10


class TestCacheDecorator:
    """缓存装饰器测试"""
    
    def test_cached_decorator(self):
        """测试缓存装饰器"""
        call_count = [0]
        
        @cached(ttl=60)
        def expensive_func(n):
            call_count[0] += 1
            return n * 2
        
        # 第一次调用
        result1 = expensive_func(5)
        assert result1 == 10
        assert call_count[0] == 1
        
        # 第二次调用（应该使用缓存）
        result2 = expensive_func(5)
        assert result2 == 10
        assert call_count[0] == 1  # 没有增加
        
        # 不同参数
        result3 = expensive_func(10)
        assert result3 == 20
        assert call_count[0] == 2


class TestAsyncTaskManager:
    """异步任务管理器测试"""
    
    def test_run_in_thread(self):
        """测试线程池执行"""
        manager = AsyncTaskManager(max_workers=2)
        
        def slow_func(n):
            time.sleep(0.1)
            return n * 2
        
        result = manager.run_in_thread(slow_func, 5)
        assert result == 10
    
    def test_batch_processor(self):
        """测试批量处理"""
        items = list(range(25))
        
        def process_item(item):
            return item * 2
        
        results = BatchProcessor.process_batch(items, process_item, batch_size=10)
        
        assert len(results) == 25
        assert results[0] == 0
        assert results[24] == 48


class TestPerformanceBenchmark:
    """性能基准测试"""
    
    def test_api_response_time(self, benchmark):
        """测试 API 响应时间"""
        def api_call():
            time.sleep(0.01)  # 模拟 API 调用
            return {"status": "ok"}
        
        result = benchmark(api_call)
        assert result["status"] == "ok"
    
    def test_cache_performance(self, benchmark):
        """测试缓存性能"""
        cache = MemoryCache()
        cache.set('test_key', 'test_value')
        
        def cache_get():
            return cache.get('test_key')
        
        result = benchmark(cache_get)
        assert result == 'test_value'
    
    def test_function_performance(self, benchmark):
        """测试函数性能"""
        @PerformanceMonitor.monitor_function
        def test_func(n):
            return sum(range(n))
        
        result = benchmark(test_func, 1000)
        assert result == sum(range(1000))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
