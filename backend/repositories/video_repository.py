"""
视频仓储
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from models.video import Video
from repositories.base import BaseRepository

class VideoRepository(BaseRepository[Video]):
    """视频仓储"""
    
    def __init__(self, db: Session):
        super().__init__(Video, db)
    
    def get_by_task(self, task_id: int) -> List[Video]:
        """获取任务的所有视频"""
        return self.db.query(Video)\
            .filter(Video.task_id == task_id)\
            .order_by(Video.scene_number)\
            .all()
    
    def get_by_scene(self, task_id: int, scene_number: int) -> Optional[Video]:
        """获取指定场景的视频"""
        return self.db.query(Video)\
            .filter(Video.task_id == task_id, Video.scene_number == scene_number)\
            .first()
    
    def count_by_task(self, task_id: int) -> int:
        """统计任务的视频数"""
        return self.db.query(Video).filter(Video.task_id == task_id).count()
