"""
任务处理器 - 协调LLM和视频生成服务
使用短会话模式：只在数据库操作时持有连接，LLM/视频生成期间释放连接，
避免 SQLite 并发写入冲突（database is locked）
"""
from utils.logger import setup_logger
from services.llm_service import generate_script
from services.video_service import generate_video_from_script
from models.database import SessionLocal
from repositories.task_repository import TaskRepository

logger = setup_logger(__name__)


def _db_update_start(task_id: str):
    """标记任务开始（独立短会话），带竞态重试"""
    import time
    for attempt in range(5):
        db = SessionLocal()
        try:
            task_repo = TaskRepository(db)
            result = task_repo.start_task(task_id)
            db.commit()
            if result is not None:
                return  # 成功
            # 任务不存在（API handler 可能还没 commit），等待后重试
            logger.warning(f"任务 {task_id} 尚未在数据库中，重试 {attempt + 1}/5")
            time.sleep(0.2)
        finally:
            db.close()
    logger.error(f"任务 {task_id} 在 5 次重试后仍然找不到")


def _db_save_script_and_update_progress(task_id: str, script: dict):
    """保存脚本到数据库并更新进度（独立短会话）"""
    db = SessionLocal()
    try:
        task_repo = TaskRepository(db)
        task_repo.update_progress(task_id, progress=30)
        db.commit()

        task = task_repo.get_by_task_id(task_id)
        if task:
            from models.script import Script
            for scene in script.get("scenes", []):
                s = Script(
                    task_id=task.id,
                    scene_number=scene.get("scene_number", 1),
                    description=scene.get("description", ""),
                    duration=scene.get("duration", 5),
                    camera_movement=scene.get("camera", ""),
                    lighting=scene.get("lighting", ""),
                )
                db.add(s)
            db.commit()
    finally:
        db.close()


def _db_complete_task(task_id: str, video_path: str):
    """标记任务完成（独立短会话）"""
    db = SessionLocal()
    try:
        task_repo = TaskRepository(db)
        task_repo.update_progress(task_id, progress=100)
        task_repo.complete_task(task_id, video_path=video_path)
        db.commit()
    finally:
        db.close()


def _db_fail_task(task_id: str, error_message: str):
    """标记任务失败（独立短会话）"""
    db = SessionLocal()
    try:
        task_repo = TaskRepository(db)
        task_repo.fail_task(task_id, error_message=error_message)
        db.commit()
    except Exception as e:
        logger.error(f"无法更新任务失败状态: {str(e)}")


def _db_update_progress(task_id: str, progress: int):
    """更新任务进度（独立短会话）"""
    db = SessionLocal()
    try:
        task_repo = TaskRepository(db)
        task_repo.update_progress(task_id, progress=progress)
        db.commit()
    finally:
        db.close()


def process_video_task(
    task_id: str,
    prompt: str,
):
    """
    处理视频生成任务（短会话模式）

    进度节点:
    -  5%: 任务开始
    - 10%: 开始生成脚本
    - 30%: 脚本生成完成
    - 50%: 开始生成视频
    - 90%: 视频生成完成（后处理前）
    - 100%: 任务完成

    Args:
        task_id: 任务ID (UUID 字符串)
        prompt: 用户输入的创作指令
    """
    try:
        logger.info(f"开始处理任务 {task_id}")

        # 阶段1: 标记任务开始 (5%)
        logger.info(f"任务 {task_id}: 标记开始")
        _db_update_start(task_id)

        # 阶段2: LLM 生成脚本 → 10% 开始，完成后 30%
        logger.info(f"任务 {task_id}: 生成脚本")
        _db_update_progress(task_id, 10)
        script = generate_script(prompt)

        # 阶段3: 保存脚本到数据库 (30%)
        logger.info(f"任务 {task_id}: 保存脚本")
        _db_save_script_and_update_progress(task_id, script)

        # 阶段4: 生成视频 → 50% 开始，视频生成完成后 90%
        logger.info(f"任务 {task_id}: 生成视频")
        _db_update_progress(task_id, 50)
        video_path = generate_video_from_script(script, task_id)

        # 阶段5: 标记任务完成 (100%)
        logger.info(f"任务 {task_id}: 标记完成")

        # 验证视频文件
        import os
        if os.path.exists(video_path):
            size_kb = os.path.getsize(video_path) / 1024
            logger.info(f"任务 {task_id} 视频文件: {video_path} ({size_kb:.1f} KB)")
        else:
            logger.error(f"任务 {task_id} 视频文件不存在: {video_path}")

        _db_complete_task(task_id, video_path)

        logger.info(f"任务 {task_id} 完成，视频路径: {video_path}")

    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {str(e)}")
        _db_fail_task(task_id, str(e))
