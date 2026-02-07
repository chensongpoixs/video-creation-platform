"""
异步处理工具模块
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AsyncTaskManager:
    """异步任务管理器"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化异步任务管理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = None  # 按需创建
    
    async def run_parallel(self, tasks: List[Callable]) -> List[Any]:
        """
        并行运行多个异步任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            结果列表
        """
        logger.info(f"并行运行 {len(tasks)} 个任务")
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查异常
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"有 {len(errors)} 个任务失败")
            
            return results
        except Exception as e:
            logger.error(f"并行任务执行失败: {str(e)}")
            raise
    
    def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """
        在线程池中运行函数
        
        Args:
            func: 要运行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数结果
        """
        logger.debug(f"在线程池中运行: {func.__name__}")
        
        future = self.thread_pool.submit(func, *args, **kwargs)
        return future.result()
    
    async def run_in_thread_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        在线程池中异步运行函数
        
        Args:
            func: 要运行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数结果
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: func(*args, **kwargs)
        )
    
    def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """
        在进程池中运行函数（CPU 密集型任务）
        
        Args:
            func: 要运行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数结果
        """
        if self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        logger.debug(f"在进程池中运行: {func.__name__}")
        
        future = self.process_pool.submit(func, *args, **kwargs)
        return future.result()
    
    def shutdown(self):
        """关闭线程池和进程池"""
        logger.info("关闭任务管理器")
        
        self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class BatchProcessor:
    """批量处理器"""
    
    @staticmethod
    def process_batch(items: List[Any], process_func: Callable, 
                     batch_size: int = 10) -> List[Any]:
        """
        批量处理项目
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数
            batch_size: 批次大小
            
        Returns:
            处理结果列表
        """
        logger.info(f"批量处理 {len(items)} 个项目，批次大小: {batch_size}")
        
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            logger.debug(f"处理批次 {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
            
            try:
                batch_results = [process_func(item) for item in batch]
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"批次处理失败: {str(e)}")
                # 继续处理下一批次
                results.extend([None] * len(batch))
        
        return results
    
    @staticmethod
    async def process_batch_async(items: List[Any], process_func: Callable,
                                  batch_size: int = 10) -> List[Any]:
        """
        异步批量处理项目
        
        Args:
            items: 要处理的项目列表
            process_func: 异步处理函数
            batch_size: 批次大小
            
        Returns:
            处理结果列表
        """
        logger.info(f"异步批量处理 {len(items)} 个项目，批次大小: {batch_size}")
        
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            logger.debug(f"处理批次 {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
            
            try:
                tasks = [process_func(item) for item in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"批次处理失败: {str(e)}")
                results.extend([None] * len(batch))
        
        return results


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        初始化速率限制器
        
        Args:
            max_calls: 时间窗口内最大调用次数
            time_window: 时间窗口（秒）
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """获取许可（异步）"""
        import time
        
        now = time.time()
        
        # 清理过期的调用记录
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        # 检查是否超过限制
        if len(self.calls) >= self.max_calls:
            # 计算需要等待的时间
            oldest_call = self.calls[0]
            wait_time = self.time_window - (now - oldest_call)
            
            if wait_time > 0:
                logger.debug(f"速率限制，等待 {wait_time:.2f}秒")
                await asyncio.sleep(wait_time)
        
        # 记录本次调用
        self.calls.append(time.time())
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器用法"""
        async def wrapper(*args, **kwargs):
            await self.acquire()
            return await func(*args, **kwargs)
        
        return wrapper


# 全局异步任务管理器
async_task_manager = AsyncTaskManager(max_workers=4)
