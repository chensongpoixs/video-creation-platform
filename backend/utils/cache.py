"""
缓存管理模块
"""
import json
import hashlib
import functools
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryCache:
    """内存缓存类"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化内存缓存
        
        Args:
            max_size: 最大缓存数量
        """
        self._cache = {}
        self._timestamps = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值或 None
        """
        if key not in self._cache:
            return None
        
        # 检查是否过期
        if key in self._timestamps:
            expire_time = self._timestamps[key]
            if datetime.now() > expire_time:
                self.delete(key)
                return None
        
        logger.debug(f"缓存命中: {key}")
        return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        # 检查缓存大小
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        self._cache[key] = value
        self._timestamps[key] = datetime.now() + timedelta(seconds=ttl)
        logger.debug(f"缓存设置: {key}, TTL: {ttl}s")
    
    def delete(self, key: str):
        """删除缓存"""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._timestamps.clear()
        logger.info("缓存已清空")
    
    def _evict_oldest(self):
        """淘汰最旧的缓存"""
        if not self._timestamps:
            return
        
        oldest_key = min(self._timestamps, key=self._timestamps.get)
        self.delete(oldest_key)
        logger.debug(f"淘汰缓存: {oldest_key}")
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'usage_percent': len(self._cache) / self._max_size * 100
        }


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, use_redis: bool = False, redis_url: str = None):
        """
        初始化缓存管理器
        
        Args:
            use_redis: 是否使用 Redis
            redis_url: Redis 连接 URL
        """
        self.memory_cache = MemoryCache()
        self.use_redis = use_redis
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(
                    redis_url or 'redis://localhost:6379'
                )
                self.redis_client.ping()
                logger.info("Redis 缓存已启用")
            except Exception as e:
                logger.warning(f"Redis 连接失败: {e}，使用内存缓存")
                self.use_redis = False
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存（多级缓存）
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值或 None
        """
        # 先查内存缓存
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # 再查 Redis
        if self.use_redis and self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    value = json.loads(cached)
                    # 回填到内存缓存
                    self.memory_cache.set(key, value)
                    logger.debug(f"Redis 缓存命中: {key}")
                    return value
            except Exception as e:
                logger.error(f"Redis 获取失败: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """
        设置缓存（多级缓存）
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        # 设置内存缓存
        self.memory_cache.set(key, value, ttl)
        
        # 设置 Redis 缓存
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.error(f"Redis 设置失败: {e}")
    
    def delete(self, key: str):
        """删除缓存"""
        self.memory_cache.delete(key)
        
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis 删除失败: {e}")
    
    def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis 清空失败: {e}")


# 全局缓存管理器
cache_manager = CacheManager()


def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    缓存装饰器
    
    Args:
        ttl: 缓存过期时间（秒）
        key_prefix: 缓存键前缀
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = _generate_cache_key(func, args, kwargs, key_prefix)
            
            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"使用缓存结果: {func.__name__}")
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


def _generate_cache_key(func: Callable, args: tuple, kwargs: dict, 
                       prefix: str = "") -> str:
    """
    生成缓存键
    
    Args:
        func: 函数
        args: 位置参数
        kwargs: 关键字参数
        prefix: 前缀
        
    Returns:
        缓存键
    """
    # 构建键字符串
    key_parts = [prefix or func.__name__]
    
    # 添加参数
    for arg in args:
        key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    key_str = ":".join(key_parts)
    
    # 如果太长，使用哈希
    if len(key_str) > 200:
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix or func.__name__}:{key_hash}"
    
    return key_str


def invalidate_cache(pattern: str = None):
    """
    清除缓存
    
    Args:
        pattern: 缓存键模式（None 清除所有）
    """
    if pattern is None:
        cache_manager.clear()
        logger.info("所有缓存已清除")
    else:
        # 简单实现：清除所有（实际应该支持模式匹配）
        cache_manager.clear()
        logger.info(f"缓存已清除: {pattern}")
