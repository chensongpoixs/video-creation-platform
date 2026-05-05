"""
任务管理API路由
"""
import os
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
from config import VIDEO_OUTPUT_DIR

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _video_url(file_path: Optional[str]) -> Optional[str]:
    """将文件系统路径转为前端可访问的 URL (/videos/xxx.mp4)"""
    if not file_path:
        return None
    filename = os.path.basename(file_path)
    return f"/videos/{filename}"


class TaskCreate(BaseModel):
    prompt: str


class SceneVideoInfo(BaseModel):
    scene_number: int
    url: str
    duration: Optional[float] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    prompt: str
    result: Optional[str] = None         # 最终拼接视频 URL
    videos: List[SceneVideoInfo] = []    # 每个分镜的独立视频
    created_at: str
    error: Optional[str] = None
    progress: Optional[int] = None


class TaskList(BaseModel):
    tasks: List[TaskResponse]
    total: int


def _scene_videos_for(task) -> List[SceneVideoInfo]:
    """构建场景视频列表：优先从 DB Video 记录读取，否则扫描视频输出目录"""
    videos = []
    seen_scenes = set()

    # 优先从数据库 Video 表读取
    if hasattr(task, 'videos') and task.videos:
        for v in sorted(task.videos, key=lambda x: x.scene_number):
            url = _video_url(v.file_path)
            if url and v.scene_number not in seen_scenes:
                videos.append(SceneVideoInfo(
                    scene_number=v.scene_number,
                    url=url,
                    duration=v.duration,
                ))
                seen_scenes.add(v.scene_number)

    # 如果 DB 中没有，扫描输出目录查找 {task_id}_scene_N.mp4
    if not videos and task.task_id:
        import glob
        import os
        pattern = os.path.join(VIDEO_OUTPUT_DIR, f"{task.task_id}_scene_*.mp4")
        scene_files = sorted(glob.glob(pattern))
        for f in scene_files:
            basename = os.path.basename(f)
            # 从 "uuid_scene_N.mp4" 提取场景号
            try:
                scene_num_str = basename.replace(f"{task.task_id}_scene_", "").replace(".mp4", "")
                scene_num = int(scene_num_str)
                if scene_num not in seen_scenes:
                    videos.append(SceneVideoInfo(
                        scene_number=scene_num,
                        url=_video_url(f),
                    ))
                    seen_scenes.add(scene_num)
            except ValueError:
                pass

    return videos


@router.post("", response_model=TaskResponse)
@router.post("/", response_model=TaskResponse)
async def create_task(
    task: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """创建新的视频生成任务"""
    print("=" * 70)
    print("[创建任务接口] 收到请求")
    print(f"  user_id: {current_user.id}")
    print(f"  username: {current_user.username}")
    print(f"  prompt: {task.prompt}")
    print("=" * 70)

    task_repo = TaskRepository(db)

    task_id = str(uuid4())

    db_task = task_repo.create(
        task_id=task_id,
        user_id=current_user.id,
        prompt=task.prompt,
        status=TaskStatus.PENDING.value,
    )

    # 显式 commit，确保后台任务启动时 task 已持久化（消除竞态）
    db.commit()

    # 添加后台任务（只传 task_id 和 prompt，后台任务自己管理 session）
    from services.task_processor import process_video_task

    background_tasks.add_task(process_video_task, task_id, task.prompt)

    print(f"[创建任务接口] ✅ 任务已创建: task_id={task_id}")

    return TaskResponse(
        task_id=db_task.task_id,
        status=db_task.status,
        prompt=db_task.prompt or "",
        result=_video_url(db_task.video_path),
        videos=_scene_videos_for(db_task),
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
    print(f"[获取任务接口] task_id={task_id}, user_id={current_user.id}")

    task_repo = TaskRepository(db)
    task = task_repo.get_with_relations(task_id)  # eager load videos + scripts

    if not task:
        print(f"[获取任务接口] ❌ 任务不存在: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.user_id != current_user.id:
        print(f"[获取任务接口] ❌ 无权访问: task.user={task.user_id}, current={current_user.id}")
        raise HTTPException(status_code=403, detail="无权访问此任务")

    print(f"[获取任务接口] ✅ 返回任务: status={task.status}")

    return TaskResponse(
        task_id=task.task_id,
        status=task.status,
        prompt=task.prompt or "",
        result=_video_url(task.video_path),
        created_at=task.created_at.isoformat() if task.created_at else "",
        error=task.error_message,
        progress=task.progress,
    )


@router.get("", response_model=TaskList)
@router.get("/", response_model=TaskList)
async def list_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取任务列表"""
    print(f"[任务列表接口] user_id={current_user.id}, skip={skip}, limit={limit}, status={status}")

    task_repo = TaskRepository(db)

    tasks = task_repo.get_by_user(
        user_id=current_user.id,
        status=status,
        skip=skip,
        limit=limit,
    )

    total = task_repo.count_by_user(current_user.id, status=status)

    print(f"[任务列表接口] ✅ 返回 {len(tasks)} 条, 共 {total} 条")

    task_responses = [
        TaskResponse(
            task_id=t.task_id,
            status=t.status,
            prompt=t.prompt or "",
            result=_video_url(t.video_path),
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
    print(f"[删除任务接口] task_id={task_id}, user_id={current_user.id}")

    task_repo = TaskRepository(db)
    task = task_repo.get_by_task_id(task_id)

    if not task:
        print(f"[删除任务接口] ❌ 任务不存在: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.user_id != current_user.id:
        print(f"[删除任务接口] ❌ 无权删除: task.user={task.user_id}, current={current_user.id}")
        raise HTTPException(status_code=403, detail="无权删除此任务")

    task_repo.delete(task.id)
    print(f"[删除任务接口] ✅ 已删除: {task_id}")
    return {"message": "任务已删除"}
