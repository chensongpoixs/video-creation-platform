"""
性能监控工具模块
"""
import time
import functools
import psutil
import threading
from typing import Dict, Any, Callable
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceMonitor:
    """性能监控类"""
    
    _metrics = {}
    _lock = threading.Lock()
    
    @classmethod
    def monitor_function(cls, func: Callable) -> Callable:
        """
        监控函数性能的装饰器
        
        Args:
            func: 要监控的函数
            
        Returns:
            包装后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # 记录性能指标
                cls._record_metric(
                    func.__name__,
                    duration,
                    memory_delta,
                    success,
                    error
                )
                
                logger.debug(
                    f"{func.__name__} - "
                    f"Time: {duration:.3f}s, "
                    f"Memory: {memory_delta:+.2f}MB, "
                    f"Success: {success}"
                )
            
            return result
        
        return wrapper
    
    @classmethod
    def _record_metric(cls, func_name: str, duration: float, 
                      memory_delta: float, success: bool, error: str = None):
        """记录性能指标"""
        with cls._lock:
            if func_name not in cls._metrics:
                cls._metrics[func_name] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0,
                    'total_memory': 0,
                    'success_count': 0,
                    'error_count': 0,
                    'errors': []
                }
            
            metrics = cls._metrics[func_name]
            metrics['count'] += 1
            metrics['total_time'] += duration
            metrics['min_time'] = min(metrics['min_time'], duration)
            metrics['max_time'] = max(metrics['max_time'], duration)
            metrics['total_memory'] += memory_delta
            
            if success:
                metrics['success_count'] += 1
            else:
                metrics['error_count'] += 1
                if error and len(metrics['errors']) < 10:
                    metrics['errors'].append(error)
    
    @classmethod
    def get_metrics(cls, func_name: str = None) -> Dict[str, Any]:
        """
        获取性能指标
        
        Args:
            func_name: 函数名（None 返回所有）
            
        Returns:
            性能指标字典
        """
        with cls._lock:
            if func_name:
                if func_name in cls._metrics:
                    metrics = cls._metrics[func_name].copy()
                    if metrics['count'] > 0:
                        metrics['avg_time'] = metrics['total_time'] / metrics['count']
                        metrics['avg_memory'] = metrics['total_memory'] / metrics['count']
                    return metrics
                return {}
            
            # 返回所有指标
            result = {}
            for name, metrics in cls._metrics.items():
                m = metrics.copy()
                if m['count'] > 0:
                    m['avg_time'] = m['total_time'] / m['count']
                    m['avg_memory'] = m['total_memory'] / m['count']
                result[name] = m
            
            return result
    
    @classmethod
    def reset_metrics(cls):
        """重置所有指标"""
        with cls._lock:
            cls._metrics.clear()
    
    @classmethod
    def get_system_metrics(cls) -> Dict[str, Any]:
        """
        获取系统资源指标
        
        Returns:
            系统指标字典
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                },
                'memory': {
                    'total': memory.total / 1024 / 1024 / 1024,  # GB
                    'available': memory.available / 1024 / 1024 / 1024,  # GB
                    'used': memory.used / 1024 / 1024 / 1024,  # GB
                    'percent': memory.percent,
                },
                'disk': {
                    'total': disk.total / 1024 / 1024 / 1024,  # GB
                    'used': disk.used / 1024 / 1024 / 1024,  # GB
                    'free': disk.free / 1024 / 1024 / 1024,  # GB
                    'percent': disk.percent,
                },
            }
            
            # GPU 信息（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    metrics['gpu'] = {
                        'available': True,
                        'count': torch.cuda.device_count(),
                        'current_device': torch.cuda.current_device(),
                        'device_name': torch.cuda.get_device_name(0),
                        'memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024 / 1024,  # GB
                        'memory_reserved': torch.cuda.memory_reserved() / 1024 / 1024 / 1024,  # GB
                    }
                else:
                    metrics['gpu'] = {'available': False}
            except ImportError:
                metrics['gpu'] = {'available': False}
            
            return metrics
            
        except Exception as e:
            logger.error(f"获取系统指标失败: {str(e)}")
            return {}
    
    @classmethod
    def print_report(cls):
        """打印性能报告"""
        metrics = cls.get_metrics()
        
        if not metrics:
            logger.info("无性能数据")
            return
        
        logger.info("=" * 80)
        logger.info("性能监控报告")
        logger.info("=" * 80)
        
        for func_name, data in sorted(metrics.items()):
            logger.info(f"\n函数: {func_name}")
            logger.info(f"  调用次数: {data['count']}")
            logger.info(f"  平均耗时: {data.get('avg_time', 0):.3f}s")
            logger.info(f"  最小耗时: {data['min_time']:.3f}s")
            logger.info(f"  最大耗时: {data['max_time']:.3f}s")
            logger.info(f"  总耗时: {data['total_time']:.3f}s")
            logger.info(f"  平均内存: {data.get('avg_memory', 0):+.2f}MB")
            logger.info(f"  成功次数: {data['success_count']}")
            logger.info(f"  失败次数: {data['error_count']}")
            
            if data['errors']:
                logger.info(f"  错误示例: {data['errors'][0]}")
        
        logger.info("=" * 80)


class ResourceMonitor:
    """资源监控类"""
    
    def __init__(self, interval: int = 60):
        """
        初始化资源监控
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.history = []
        self.max_history = 1000
    
    def start(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"资源监控已启动，间隔 {self.interval}秒")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("资源监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = PerformanceMonitor.get_system_metrics()
                self.history.append(metrics)
                
                # 限制历史记录数量
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # 检查资源警告
                self._check_warnings(metrics)
                
            except Exception as e:
                logger.error(f"资源监控错误: {str(e)}")
            
            time.sleep(self.interval)
    
    def _check_warnings(self, metrics: Dict[str, Any]):
        """检查资源警告"""
        # CPU 警告
        if metrics.get('cpu', {}).get('percent', 0) > 80:
            logger.warning(f"CPU 使用率过高: {metrics['cpu']['percent']:.1f}%")
        
        # 内存警告
        if metrics.get('memory', {}).get('percent', 0) > 80:
            logger.warning(f"内存使用率过高: {metrics['memory']['percent']:.1f}%")
        
        # 磁盘警告
        if metrics.get('disk', {}).get('percent', 0) > 90:
            logger.warning(f"磁盘使用率过高: {metrics['disk']['percent']:.1f}%")
        
        # GPU 内存警告
        gpu = metrics.get('gpu', {})
        if gpu.get('available') and gpu.get('memory_allocated', 0) > 10:
            logger.warning(f"GPU 内存占用过高: {gpu['memory_allocated']:.2f}GB")
    
    def get_history(self, limit: int = None) -> list:
        """
        获取历史记录
        
        Args:
            limit: 限制数量
            
        Returns:
            历史记录列表
        """
        if limit:
            return self.history[-limit:]
        return self.history.copy()
    
    def get_average_metrics(self) -> Dict[str, Any]:
        """获取平均指标"""
        if not self.history:
            return {}
        
        cpu_avg = sum(m.get('cpu', {}).get('percent', 0) for m in self.history) / len(self.history)
        memory_avg = sum(m.get('memory', {}).get('percent', 0) for m in self.history) / len(self.history)
        
        return {
            'cpu_percent_avg': cpu_avg,
            'memory_percent_avg': memory_avg,
            'sample_count': len(self.history),
        }


# 全局资源监控实例
resource_monitor = ResourceMonitor(interval=60)
