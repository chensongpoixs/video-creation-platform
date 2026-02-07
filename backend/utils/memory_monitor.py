"""
显存监控工具 - 监控和管理 GPU 显存使用
"""
import torch
import time
from typing import Dict, Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MemoryMonitor:
    """GPU 显存监控器"""
    
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.enabled else 0
        self.history = []
        
        if self.enabled:
            logger.info(f"显存监控器初始化，检测到 {self.device_count} 个 GPU")
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, 总显存: {props.total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA 不可用，显存监控器禁用")
    
    def get_memory_info(self, device: int = 0) -> Dict[str, float]:
        """
        获取显存信息
        
        Args:
            device: GPU 设备编号
            
        Returns:
            显存信息字典 (单位: GB)
        """
        if not self.enabled:
            return {
                "total": 0.0,
                "allocated": 0.0,
                "reserved": 0.0,
                "free": 0.0,
                "usage_percent": 0.0
            }
        
        try:
            props = torch.cuda.get_device_properties(device)
            total = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            free = total - allocated
            usage_percent = (allocated / total) * 100 if total > 0 else 0
            
            return {
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "free": free,
                "usage_percent": usage_percent
            }
        except Exception as e:
            logger.error(f"获取显存信息失败: {str(e)}")
            return {}
    
    def print_memory_info(self, device: int = 0, prefix: str = ""):
        """
        打印显存信息
        
        Args:
            device: GPU 设备编号
            prefix: 输出前缀
        """
        info = self.get_memory_info(device)
        if not info:
            return
        
        msg = f"{prefix}显存使用: {info['allocated']:.2f}/{info['total']:.2f} GB ({info['usage_percent']:.1f}%)"
        logger.info(msg)
        print(msg)
    
    def check_available_memory(self, required_gb: float, device: int = 0) -> bool:
        """
        检查是否有足够的可用显存
        
        Args:
            required_gb: 需要的显存大小 (GB)
            device: GPU 设备编号
            
        Returns:
            是否有足够显存
        """
        if not self.enabled:
            logger.warning("CUDA 不可用，跳过显存检查")
            return False
        
        info = self.get_memory_info(device)
        available = info.get("free", 0)
        
        if available >= required_gb:
            logger.info(f"显存检查通过: 需要 {required_gb:.2f} GB, 可用 {available:.2f} GB")
            return True
        else:
            logger.warning(f"显存不足: 需要 {required_gb:.2f} GB, 可用 {available:.2f} GB")
            return False
    
    def clear_cache(self, device: Optional[int] = None):
        """
        清理显存缓存
        
        Args:
            device: GPU 设备编号，None 表示所有设备
        """
        if not self.enabled:
            return
        
        try:
            before = self.get_memory_info(device or 0)
            torch.cuda.empty_cache()
            after = self.get_memory_info(device or 0)
            
            freed = before.get("reserved", 0) - after.get("reserved", 0)
            if freed > 0:
                logger.info(f"清理显存缓存，释放 {freed:.2f} GB")
            else:
                logger.debug("显存缓存已清理")
        except Exception as e:
            logger.error(f"清理显存缓存失败: {str(e)}")
    
    def record_snapshot(self, label: str = "", device: int = 0):
        """
        记录显存快照
        
        Args:
            label: 快照标签
            device: GPU 设备编号
        """
        info = self.get_memory_info(device)
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "device": device,
            **info
        }
        self.history.append(snapshot)
        logger.debug(f"记录显存快照: {label}, 已分配 {info.get('allocated', 0):.2f} GB")
    
    def get_peak_memory(self, device: int = 0) -> float:
        """
        获取峰值显存使用
        
        Args:
            device: GPU 设备编号
            
        Returns:
            峰值显存 (GB)
        """
        if not self.enabled:
            return 0.0
        
        try:
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            return peak
        except Exception as e:
            logger.error(f"获取峰值显存失败: {str(e)}")
            return 0.0
    
    def reset_peak_memory(self, device: int = 0):
        """
        重置峰值显存统计
        
        Args:
            device: GPU 设备编号
        """
        if not self.enabled:
            return
        
        try:
            torch.cuda.reset_peak_memory_stats(device)
            logger.debug(f"重置 GPU {device} 峰值显存统计")
        except Exception as e:
            logger.error(f"重置峰值显存统计失败: {str(e)}")
    
    def suggest_optimization(self, device: int = 0) -> Dict[str, any]:
        """
        根据当前显存使用情况提供优化建议
        
        Args:
            device: GPU 设备编号
            
        Returns:
            优化建议字典
        """
        info = self.get_memory_info(device)
        suggestions = {
            "use_fp16": False,
            "reduce_batch_size": False,
            "enable_gradient_checkpointing": False,
            "reduce_resolution": False,
            "message": ""
        }
        
        usage = info.get("usage_percent", 0)
        free = info.get("free", 0)
        
        if usage > 90:
            suggestions["use_fp16"] = True
            suggestions["reduce_batch_size"] = True
            suggestions["message"] = "显存使用率超过 90%，建议启用 FP16 并减少批次大小"
        elif usage > 80:
            suggestions["use_fp16"] = True
            suggestions["message"] = "显存使用率超过 80%，建议启用 FP16"
        elif free < 2.0:
            suggestions["reduce_resolution"] = True
            suggestions["message"] = "可用显存不足 2GB，建议降低分辨率"
        else:
            suggestions["message"] = "显存使用正常"
        
        return suggestions
    
    def get_memory_summary(self) -> str:
        """
        获取显存使用摘要
        
        Returns:
            显存摘要字符串
        """
        if not self.enabled:
            return "CUDA 不可用"
        
        summary_lines = []
        for i in range(self.device_count):
            info = self.get_memory_info(i)
            props = torch.cuda.get_device_properties(i)
            
            summary_lines.append(f"GPU {i} ({props.name}):")
            summary_lines.append(f"  总显存: {info['total']:.2f} GB")
            summary_lines.append(f"  已分配: {info['allocated']:.2f} GB")
            summary_lines.append(f"  已保留: {info['reserved']:.2f} GB")
            summary_lines.append(f"  可用: {info['free']:.2f} GB")
            summary_lines.append(f"  使用率: {info['usage_percent']:.1f}%")
            
            peak = self.get_peak_memory(i)
            summary_lines.append(f"  峰值: {peak:.2f} GB")
        
        return "\n".join(summary_lines)

# 全局实例
memory_monitor = MemoryMonitor()

def print_memory(prefix: str = ""):
    """便捷函数：打印显存信息"""
    memory_monitor.print_memory_info(prefix=prefix)

def clear_memory():
    """便捷函数：清理显存"""
    memory_monitor.clear_cache()

def check_memory(required_gb: float) -> bool:
    """便捷函数：检查显存"""
    return memory_monitor.check_available_memory(required_gb)
