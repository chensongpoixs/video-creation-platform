"""
性能监控中间件
"""
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logger import setup_logger
from utils.performance import PerformanceMonitor

logger = setup_logger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """性能监控中间件"""
    
    async def dispatch(self, request: Request, call_next):
        """
        处理请求并监控性能
        
        Args:
            request: 请求对象
            call_next: 下一个中间件
            
        Returns:
            响应对象
        """
        # 记录开始时间
        start_time = time.time()
        
        # 获取请求信息
        method = request.method
        path = request.url.path
        client = request.client.host if request.client else "unknown"
        
        # 处理请求
        try:
            response = await call_next(request)
            status_code = response.status_code
            success = True
        except Exception as e:
            logger.error(f"请求处理失败: {str(e)}")
            status_code = 500
            success = False
            raise
        finally:
            # 计算耗时
            duration = time.time() - start_time
            
            # 记录日志
            log_message = (
                f"{method} {path} - "
                f"Status: {status_code}, "
                f"Duration: {duration:.3f}s, "
                f"Client: {client}"
            )
            
            if duration > 1.0:
                logger.warning(f"慢请求: {log_message}")
            else:
                logger.info(log_message)
            
            # 添加性能头
            if success:
                response.headers["X-Process-Time"] = f"{duration:.3f}"
        
        return response


class RequestCounterMiddleware(BaseHTTPMiddleware):
    """请求计数中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
    
    async def dispatch(self, request: Request, call_next):
        """处理请求并计数"""
        self.request_count += 1
        
        try:
            response = await call_next(request)
            
            if response.status_code >= 400:
                self.error_count += 1
            
            return response
        except Exception as e:
            self.error_count += 1
            raise
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
        }
