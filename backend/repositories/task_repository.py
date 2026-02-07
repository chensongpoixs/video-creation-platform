"""
任务仓储
"""
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc, func
from datetime import datetime
from models.task import Task, TaskStatus
from repositories.base import BaseRepository

class TaskRepository(BaseRepository[Task]):
    """任务仓储"""
    
    def __init__(self, db: Session):
        super().__init__(Task, db)
    
    def get_by_task_id(self, task_id: str) -> Optional[Task]:
        """根据任务ID获取"""
        return self.db.query(Task).filter(Task.task_id == task_id).first()
    
    def get_with_relations(self, task_id: str) -> Optional[Task]:
        """获取任务及其关联数据"""
        return self.db.query(Task)\
            .options(joinedload(Task.scripts))\
            .options(joinedload(Task.videos))\
            .filter(Task.task_id == task_id)\
            .first()
    
    def get_by_user(self, user_id: int, status: Optional[str] = None, 
                    skip: int = 0, limit: int = 10) -> List[Task]:
        """获取用户的任务列表"""
        query = self.db.query(Task).filter(Task.user_id == user_id)
        
        if status:
            query = query.filter(Task.status == status)
        
        return query.order_by(desc(Task.created_at))\
            .offset(skip)\
            .limit(limit)\
            .all()
    
    def get_recent(self, limit: int = 20) -> List[Task]:
        """获取最近的任务"""
        return self.db.query(Task)\
            .order_by(desc(Task.created_at))\
            .limit(limit)\
            .all()
    
    def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[Task]:
        """根据状态获取任务"""
        return self.db.query(Task)\
            .filter(Task.status == status)\
            .order_by(desc(Task.created_at))\
            .offset(skip)\
            .limit(limit)\
            .all()
    
    def count_by_status(self, status: str) -> int:
        """统计指定状态的任务数"""
        return self.db.query(Task).filter(Task.status == status).count()
    
    def count_by_user(self, user_id: int, status: Optional[str] = None) -> int:
        """统计用户的任务数"""
        query = self.db.query(Task).filter(Task.user_id == user_id)
        if status:
            query = query.filter(Task.status == status)
        return query.count()
    
    def update_status(self, task_id: str, status: str) -> Optional[Task]:
        """更新任务状态"""
        task = self.get_by_task_id(task_id)
        if task:
            task.status = status
            self.db.flush()
            self.db.refresh(task)
        return task
    
    def update_progress(self, task_id: str, progress: int, 
                       completed_scenes: Optional[int] = None) -> Optional[Task]:
        """更新任务进度"""
        task = self.get_by_task_id(task_id)
        if task:
            task.progress = progress
            if completed_scenes is not None:
                task.completed_scenes = completed_scenes
            self.db.flush()
            self.db.refresh(task)
        return task
    
    def start_task(self, task_id: str) -> Optional[Task]:
        """开始任务"""
        task = self.get_by_task_id(task_id)
        if task:
            task.start()
            self.db.flush()
            self.db.refresh(task)
        return task
    
    def complete_task(self, task_id: str, video_path: Optional[str] = None) -> Optional[Task]:
        """完成任务"""
        task = self.get_by_task_id(task_id)
        if task:
            task.complete(video_path)
            self.db.flush()
            self.db.refresh(task)
        return task
    
    def fail_task(self, task_id: str, error_message: str) -> Optional[Task]:
        """任务失败"""
        task = self.get_by_task_id(task_id)
        if task:
            task.fail(error_message)
            self.db.flush()
            self.db.refresh(task)
        return task
    
    def get_statistics(self, date_from: Optional[datetime] = None, 
                      date_to: Optional[datetime] = None) -> dict:
        """获取任务统计"""
        query = self.db.query(Task)
        
        if date_from:
            query = query.filter(Task.created_at >= date_from)
        if date_to:
            query = query.filter(Task.created_at <= date_to)
        
        total = query.count()
        completed = query.filter(Task.status == TaskStatus.COMPLETED.value).count()
        failed = query.filter(Task.status == TaskStatus.FAILED.value).count()
        processing = query.filter(Task.status == TaskStatus.PROCESSING.value).count()
        
        # 平均生成时间
        avg_duration = query.filter(Task.duration.isnot(None))\
            .with_entities(func.avg(Task.duration))\
            .scalar() or 0
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "success_rate": round((completed / total * 100) if total > 0 else 0, 2),
            "avg_duration": round(avg_duration, 2)
        }
