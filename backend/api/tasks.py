"""
任务管理API路由
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from uuid import uuid4
from datetime import datetime
from sqlalchemy.orm import Session

from models.database import get_db
from models.task import TaskStatus
from repositories.task_repository import TaskRepository
from middleware.auth_middleware import get_current_active_user
from models.user import User

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class TaskCreate(BaseModel):
    prompt: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    prompt: str
    result: Optional[str] = None
    created_at: str
    error: Optional[str] = None
    progress: Optional[int] = None


class TaskList(BaseModel):
    tasks: List[TaskResponse]
    total: int


@router.post("/", response_model=TaskResponse)
async def create_task(
    task: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """创建新的视频生成任务"""
    task_repo = TaskRepository(db)

    task_id = str(uuid4())

    db_task = task_repo.create(
        task_id=task_id,
        user_id=current_user.id,
        prompt=task.prompt,
        status=TaskStatus.PENDING.value,
    )

    # 添加后台任务
    from services.task_processor import process_video_task

    background_tasks.add_task(process_video_task, task_id, task.prompt, task_repo, db)

    return TaskResponse(
        task_id=db_task.task_id,
        status=db_task.status,
        prompt=db_task.prompt,
        result=db_task.video_path,
        created_at=db_task.created_at.isoformat() if db_task.created_at else datetime.now().isoformat(),
        error=db_task.error_message,
        progress=db_task.progress,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取任务详情"""
    task_repo = TaskRepository(db)
    task = task_repo.get_by_task_id(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问此任务")

    return TaskResponse(
        task_id=task.task_id,
        status=task.status,
        prompt=task.prompt or "",
        result=task.video_path,
        created_at=task.created_at.isoformat() if task.created_at else "",
        error=task.error_message,
        progress=task.progress,
    )


@router.get("/", response_model=TaskList)
async def list_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取任务列表"""
    task_repo = TaskRepository(db)

    tasks = task_repo.get_by_user(
        user_id=current_user.id,
        status=status,
        skip=skip,
        limit=limit,
    )

    total = task_repo.count_by_user(current_user.id, status=status)

    task_responses = [
        TaskResponse(
            task_id=t.task_id,
            status=t.status,
            prompt=t.prompt or "",
            result=t.video_path,
            created_at=t.created_at.isoformat() if t.created_at else "",
            error=t.error_message,
            progress=t.progress,
        )
        for t in tasks
    ]

    return TaskList(tasks=task_responses, total=total)


@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """删除任务"""
    task_repo = TaskRepository(db)
    task = task_repo.get_by_task_id(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权删除此任务")

    task_repo.delete(task.id)
    return {"message": "任务已删除"}
