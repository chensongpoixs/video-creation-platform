"""
任务处理器 - 协调LLM和视频生成服务
"""
from utils.logger import setup_logger
from services.llm_service import generate_script
from services.video_service import generate_video_from_script

logger = setup_logger(__name__)

def process_video_task(task_id: str, prompt: str, tasks_db: dict):
    """
    处理视频生成任务
    
    Args:
        task_id: 任务ID
        prompt: 用户输入的创作指令
        tasks_db: 任务数据库引用
    """
    try:
        logger.info(f"开始处理任务 {task_id}")
        tasks_db[task_id]['status'] = 'processing'
        
        # 步骤1: 生成脚本和分镜
        logger.info(f"任务 {task_id}: 生成脚本")
        script = generate_script(prompt)
        tasks_db[task_id]['script'] = script
        
        # 步骤2: 生成视频
        logger.info(f"任务 {task_id}: 生成视频")
        video_path = generate_video_from_script(script, task_id)
        
        # 步骤3: 更新任务状态
        tasks_db[task_id]['status'] = 'completed'
        tasks_db[task_id]['result'] = video_path
        logger.info(f"任务 {task_id} 完成，视频路径: {video_path}")
        
    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {str(e)}")
        tasks_db[task_id]['status'] = 'failed'
        tasks_db[task_id]['error'] = str(e)
