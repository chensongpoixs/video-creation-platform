"""
任务管理API路由
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from uuid import uuid4
from datetime import datetime

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

# 任务存储（生产环境应使用数据库）
tasks_db = {}

class TaskCreate(BaseModel):
    prompt: str
    
class TaskResponse(BaseModel):
    task_id: str
    status: str
    prompt: str
    result: Optional[str] = None
    created_at: str
    error: Optional[str] = None

class TaskList(BaseModel):
    tasks: List[TaskResponse]
    total: int

@router.post("/", response_model=TaskResponse)
async def create_task(task: TaskCreate, background_tasks: BackgroundTasks):
    """创建新的视频生成任务"""
    task_id = str(uuid4())
    
    task_data = {
        "task_id": task_id,
        "status": "pending",
        "prompt": task.prompt,
        "result": None,
        "created_at": datetime.now().isoformat(),
        "error": None
    }
    
    tasks_db[task_id] = task_data
    
    # 添加后台任务
    from services.task_processor import process_video_task
    background_tasks.add_task(process_video_task, task_id, task.prompt, tasks_db)
    
    return TaskResponse(**task_data)

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """获取任务详情"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return TaskResponse(**tasks_db[task_id])

@router.get("/", response_model=TaskList)
async def list_tasks(skip: int = 0, limit: int = 10):
    """获取任务列表"""
    all_tasks = list(tasks_db.values())
    total = len(all_tasks)
    tasks = all_tasks[skip:skip + limit]
    
    return TaskList(
        tasks=[TaskResponse(**t) for t in tasks],
        total=total
    )

@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    del tasks_db[task_id]
    return {"message": "任务已删除"}
