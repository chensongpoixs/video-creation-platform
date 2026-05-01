"""
任务处理器 - 协调LLM和视频生成服务
"""
from sqlalchemy.orm import Session
from utils.logger import setup_logger
from services.llm_service import generate_script
from services.video_service import generate_video_from_script
from repositories.task_repository import TaskRepository

logger = setup_logger(__name__)


def process_video_task(
    task_id: str,
    prompt: str,
    task_repo: TaskRepository,
    db: Session,
):
    """
    处理视频生成任务（数据库持久化版本）

    Args:
        task_id: 任务ID (UUID 字符串)
        prompt: 用户输入的创作指令
        task_repo: 任务仓储实例
        db: 数据库会话
    """
    try:
        logger.info(f"开始处理任务 {task_id}")

        # 开始任务
        task_repo.start_task(task_id)

        # 步骤1: 生成脚本和分镜
        logger.info(f"任务 {task_id}: 生成脚本")
        script = generate_script(prompt)
        task_repo.update_progress(task_id, progress=30)

        # 保存脚本到数据库
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

        # 步骤2: 生成视频
        logger.info(f"任务 {task_id}: 生成视频")
        task_repo.update_progress(task_id, progress=50)
        video_path = generate_video_from_script(script, task_id)

        # 步骤3: 完成任务
        task_repo.update_progress(task_id, progress=100)
        task_repo.complete_task(task_id, video_path=video_path)
        logger.info(f"任务 {task_id} 完成，视频路径: {video_path}")

    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {str(e)}")
        try:
            task_repo.fail_task(task_id, error_message=str(e))
        except Exception as inner:
            logger.error(f"无法更新任务失败状态: {str(inner)}")
