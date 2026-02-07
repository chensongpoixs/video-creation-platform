"""
视频生成服务模块 - 负责视频生成和后处理
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from utils.logger import setup_logger
from config import VIDEO_OUTPUT_DIR, VIDEO_CONFIG, VIDEO_POST_PROCESSING_CONFIG
from services.video_processor import VideoProcessor

logger = setup_logger(__name__)

def generate_video_from_script(script: Dict, task_id: str, 
                               post_process_config: Optional[Dict] = None) -> str:
    """
    根据脚本生成完整视频（含后处理）
    
    Args:
        script: 包含分镜信息的脚本字典
        task_id: 任务ID
        post_process_config: 后处理配置（可选，默认使用全局配置）
        
    Returns:
        生成的视频文件路径
    """
    try:
        logger.info(f"开始生成视频，任务ID: {task_id}")
        
        # 使用默认配置或自定义配置
        if post_process_config is None:
            post_process_config = VIDEO_POST_PROCESSING_CONFIG
        
        video_paths = []
        
        # 为每个分镜生成视频片段
        for scene in script['scenes']:
            logger.info(f"生成场景 {scene['scene_number']}: {scene['description'][:50]}")
            video_path = generate_scene_video(scene, task_id)
            video_paths.append(video_path)
        
        # 拼接所有视频片段
        stitched_video = stitch_videos(video_paths, task_id)
        
        # 应用后处理
        final_video = apply_post_processing(
            stitched_video, task_id, post_process_config
        )
        
        logger.info(f"✅ 视频生成完成: {final_video}")
        return final_video
        
    except Exception as e:
        logger.error(f"视频生成失败: {str(e)}")
        raise

def generate_scene_video(scene: Dict, task_id: str) -> str:
    """
    生成单个场景的视频片段
    
    Args:
        scene: 场景信息字典
        task_id: 任务ID
        
    Returns:
        视频片段文件路径
    """
    try:
        from services.model_loader import video_loader
        
        scene_id = scene['scene_number']
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_scene_{scene_id}.mp4")
        
        # 检查模型是否加载
        if not video_loader.is_loaded:
            logger.warning("视频模型未加载，使用备用方案")
            return generate_scene_video_fallback(scene, task_id)
        
        # 优化提示词
        from services.llm_service import optimize_prompt_for_video
        prompt = optimize_prompt_for_video(scene['description'])
        
        logger.info(f"生成提示词: {prompt[:60]}...")
        
        # 生成占位符图像
        image = VideoProcessor.generate_placeholder_image(
            width=VIDEO_CONFIG.get("width", 1024),
            height=VIDEO_CONFIG.get("height", 576)
        )
        
        # 生成视频帧
        logger.info("调用视频模型生成帧...")
        frames = video_loader.generate_video(
            prompt=prompt,
            image=image,
            num_frames=VIDEO_CONFIG.get("num_frames", 25)
        )
        
        # 帧插值（可选）
        if VIDEO_CONFIG.get("enable_interpolation", False):
            frames = VideoProcessor.interpolate_frames(frames, factor=2)
        
        # 转换为视频文件
        fps = VIDEO_CONFIG.get("fps", 6)
        VideoProcessor.frames_to_video(frames, output_path, fps=fps)
        
        return output_path
        
    except Exception as e:
        logger.error(f"场景视频生成失败: {str(e)}")
        # 使用备用方案
        return generate_scene_video_fallback(scene, task_id)

def generate_scene_video_fallback(scene: Dict, task_id: str) -> str:
    """
    生成场景视频的备用方案（当模型不可用时）
    """
    logger.info(f"使用备用方案生成场景 {scene['scene_number']}")
    
    scene_id = scene['scene_number']
    duration = scene['duration']
    fps = VIDEO_CONFIG.get("fps", 6)
    width = VIDEO_CONFIG.get("width", 1024)
    height = VIDEO_CONFIG.get("height", 576)
    
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_scene_{scene_id}.mp4")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 生成随机颜色的帧
    color = np.random.randint(0, 255, 3).tolist()
    total_frames = duration * fps
    
    for i in range(total_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        
        # 添加场景描述文字
        text = f"Scene {scene_id}: {scene['description'][:40]}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 添加进度信息
        progress_text = f"Frame {i+1}/{total_frames}"
        cv2.putText(frame, progress_text, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        out.write(frame)
    
    out.release()
    logger.info(f"✅ 备用视频生成完成: {output_path}")
    return output_path

def stitch_videos(video_paths: List[str], task_id: str) -> str:
    """
    拼接多个视频片段
    
    Args:
        video_paths: 视频文件路径列表
        task_id: 任务ID
        
    Returns:
        拼接后的视频文件路径
    """
    if not video_paths:
        raise ValueError("没有视频片段可拼接")
    
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_final.mp4")
    
    logger.info(f"开始拼接视频，片段数: {len(video_paths)}")
    
    # 读取第一个视频获取参数
    cap = cv2.VideoCapture(video_paths[0])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    logger.info(f"视频参数: {width}x{height}, {fps} FPS")
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 逐个读取并写入视频片段
    for i, video_path in enumerate(video_paths):
        logger.info(f"拼接片段 {i+1}/{len(video_paths)}: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        
        logger.info(f"片段 {i+1} 拼接完成，共 {frame_count} 帧")
        cap.release()
    
    out.release()
    
    logger.info(f"✅ 视频拼接完成: {output_path}")
    return output_path

def add_subtitles(video_path: str, subtitles: List[dict]) -> str:
    """
    为视频添加字幕
    
    Args:
        video_path: 视频文件路径
        subtitles: 字幕列表 [{"text": "...", "start": 0, "duration": 5}, ...]
        
    Returns:
        添加字幕后的视频路径
    """
    logger.warning("字幕功能待实现")
    pass


def apply_post_processing(video_path: str, task_id: str, 
                         config: Dict) -> str:
    """
    应用视频后处理
    
    Args:
        video_path: 输入视频路径
        task_id: 任务ID
        config: 后处理配置
        
    Returns:
        处理后的视频路径
    """
    logger.info("开始视频后处理")
    
    current_video = video_path
    temp_dir = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. 应用滤镜
        if config.get('filters'):
            filters = config['filters']
            if any([
                filters.get('brightness', 1.0) != 1.0,
                filters.get('contrast', 1.0) != 1.0,
                filters.get('saturation', 1.0) != 1.0,
                filters.get('sharpen', False),
                filters.get('blur', 0) > 0,
                filters.get('vignette', 0) > 0,
                filters.get('sepia', False),
                filters.get('grayscale', False),
            ]):
                logger.info("应用滤镜...")
                from services.video_filter import VideoFilter
                filtered_video = os.path.join(temp_dir, "filtered.mp4")
                current_video = VideoFilter.apply_filter_to_video(
                    current_video, filtered_video, filters
                )
        
        # 2. 添加字幕
        if config.get('subtitle', {}).get('enabled'):
            logger.info("添加字幕...")
            from services.subtitle_system import SubtitleSystem
            
            subtitle_config = config['subtitle']
            subtitle_system = SubtitleSystem(
                font_path=subtitle_config.get('font_path'),
                font_size=subtitle_config.get('font_size', 32),
                font_color=subtitle_config.get('font_color', (255, 255, 255)),
                bg_color=subtitle_config.get('bg_color'),
                position=subtitle_config.get('position', 'bottom')
            )
            
            # 这里需要字幕数据，暂时跳过
            # subtitles = subtitle_config.get('subtitles', [])
            # if subtitles:
            #     subtitled_video = os.path.join(temp_dir, "subtitled.mp4")
            #     current_video = subtitle_system.add_subtitles_to_video(
            #         current_video, subtitled_video, subtitles
            #     )
        
        # 3. 添加背景音乐
        if config.get('audio', {}).get('background_music'):
            logger.info("添加背景音乐...")
            from services.audio_processor import AudioProcessor
            
            audio_config = config['audio']
            music_path = audio_config['background_music']
            
            if os.path.exists(music_path):
                audio_video = os.path.join(temp_dir, "with_audio.mp4")
                current_video = AudioProcessor.add_background_music(
                    current_video, music_path, audio_video,
                    volume=audio_config.get('volume', 0.3)
                )
        
        # 4. 质量优化
        if config.get('optimization'):
            opt_config = config['optimization']
            if any([
                opt_config.get('denoise', False),
                opt_config.get('color_correction', False),
                opt_config.get('enhance_contrast', False),
            ]):
                logger.info("质量优化...")
                from services.video_optimizer import VideoOptimizer
                
                optimized_video = os.path.join(temp_dir, "optimized.mp4")
                current_video = VideoOptimizer.optimize_video(
                    current_video, optimized_video,
                    denoise=opt_config.get('denoise', False),
                    color_correct=opt_config.get('color_correction', False),
                    enhance_contrast=opt_config.get('enhance_contrast', False)
                )
        
        # 5. 格式转换和压缩
        if config.get('output', {}).get('compress', True):
            logger.info("压缩视频...")
            from services.video_converter import VideoConverter
            
            output_config = config['output']
            final_video = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_final.mp4")
            
            current_video = VideoConverter.compress_video(
                current_video, final_video,
                crf=output_config.get('crf', 23),
                preset=output_config.get('preset', 'medium')
            )
        else:
            # 不压缩，直接重命名
            final_video = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_final.mp4")
            import shutil
            shutil.copy2(current_video, final_video)
            current_video = final_video
        
        logger.info(f"✅ 后处理完成: {current_video}")
        return current_video
        
    except Exception as e:
        logger.error(f"后处理失败: {str(e)}")
        # 返回原始视频
        return video_path
    finally:
        # 清理临时文件
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass
