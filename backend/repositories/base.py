"""
基础仓储类
"""
from typing import TypeVar, Generic, Type, List, Optional
from sqlalchemy.orm import Session
from models.database import Base

ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    """基础仓储类"""
    
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db
    
    def get(self, id: int) -> Optional[ModelType]:
        """根据ID获取"""
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """获取所有记录"""
        return self.db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, **kwargs) -> ModelType:
        """创建记录"""
        obj = self.model(**kwargs)
        self.db.add(obj)
        self.db.flush()
        self.db.refresh(obj)
        return obj
    
    def update(self, id: int, **kwargs) -> Optional[ModelType]:
        """更新记录"""
        obj = self.get(id)
        if obj:
            for key, value in kwargs.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            self.db.flush()
            self.db.refresh(obj)
        return obj
    
    def delete(self, id: int) -> bool:
        """删除记录"""
        obj = self.get(id)
        if obj:
            self.db.delete(obj)
            self.db.flush()
            return True
        return False
    
    def count(self) -> int:
        """统计总数"""
        return self.db.query(self.model).count()
    
    def exists(self, id: int) -> bool:
        """检查是否存在"""
        return self.db.query(self.model).filter(self.model.id == id).count() > 0
