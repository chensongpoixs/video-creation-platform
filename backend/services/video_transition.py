"""
视频转场效果模块
"""
import cv2
import numpy as np
from typing import List
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TransitionEffect:
    """转场效果处理类"""
    
    @staticmethod
    def fade_transition(video1_path: str, video2_path: str, 
                       output_path: str, duration_frames: int = 15) -> str:
        """
        淡入淡出转场
        
        Args:
            video1_path: 第一个视频路径
            video2_path: 第二个视频路径
            output_path: 输出视频路径
            duration_frames: 转场持续帧数
            
        Returns:
            输出视频路径
        """
        logger.info(f"应用淡入淡出转场，持续 {duration_frames} 帧")
        
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 写入第一个视频（除了最后 duration_frames 帧）
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frames1 = []
        
        for i in range(total_frames1):
            ret, frame = cap1.read()
            if not ret:
                break
            
            if i < total_frames1 - duration_frames:
                out.write(frame)
            else:
                frames1.append(frame)
        
        # 读取第二个视频的前 duration_frames 帧
        frames2 = []
        for i in range(duration_frames):
            ret, frame = cap2.read()
            if not ret:
                break
            frames2.append(frame)
        
        # 创建转场帧
        for i in range(min(len(frames1), len(frames2))):
            alpha = (i + 1) / len(frames1)
            blended = cv2.addWeighted(frames1[i], 1 - alpha, frames2[i], alpha, 0)
            out.write(blended)
        
        # 写入第二个视频的剩余部分
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            out.write(frame)
        
        cap1.release()
        cap2.release()
        out.release()
        
        logger.info(f"✅ 淡入淡出转场完成: {output_path}")
        return output_path
    
    @staticmethod
    def cross_dissolve(video1_path: str, video2_path: str, 
                      output_path: str, duration_frames: int = 15) -> str:
        """
        交叉溶解转场（与淡入淡出类似，但更平滑）
        
        Args:
            video1_path: 第一个视频路径
            video2_path: 第二个视频路径
            output_path: 输出视频路径
            duration_frames: 转场持续帧数
            
        Returns:
            输出视频路径
        """
        # 使用 S 曲线进行更平滑的过渡
        logger.info(f"应用交叉溶解转场，持续 {duration_frames} 帧")
        
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 写入第一个视频（除了最后 duration_frames 帧）
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frames1 = []
        
        for i in range(total_frames1):
            ret, frame = cap1.read()
            if not ret:
                break
            
            if i < total_frames1 - duration_frames:
                out.write(frame)
            else:
                frames1.append(frame)
        
        # 读取第二个视频的前 duration_frames 帧
        frames2 = []
        for i in range(duration_frames):
            ret, frame = cap2.read()
            if not ret:
                break
            frames2.append(frame)
        
        # 创建转场帧（使用 S 曲线）
        for i in range(min(len(frames1), len(frames2))):
            t = (i + 1) / len(frames1)
            # S 曲线: 3t^2 - 2t^3
            alpha = 3 * t * t - 2 * t * t * t
            blended = cv2.addWeighted(frames1[i], 1 - alpha, frames2[i], alpha, 0)
            out.write(blended)
        
        # 写入第二个视频的剩余部分
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            out.write(frame)
        
        cap1.release()
        cap2.release()
        out.release()
        
        logger.info(f"✅ 交叉溶解转场完成: {output_path}")
        return output_path
    
    @staticmethod
    def slide_transition(video1_path: str, video2_path: str, 
                        output_path: str, direction: str = 'left',
                        duration_frames: int = 15) -> str:
        """
        滑动转场
        
        Args:
            video1_path: 第一个视频路径
            video2_path: 第二个视频路径
            output_path: 输出视频路径
            direction: 滑动方向 ('left', 'right', 'up', 'down')
            duration_frames: 转场持续帧数
            
        Returns:
            输出视频路径
        """
        logger.info(f"应用滑动转场（{direction}），持续 {duration_frames} 帧")
        
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 写入第一个视频（除了最后 duration_frames 帧）
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frames1 = []
        
        for i in range(total_frames1):
            ret, frame = cap1.read()
            if not ret:
                break
            
            if i < total_frames1 - duration_frames:
                out.write(frame)
            else:
                frames1.append(frame)
        
        # 读取第二个视频的前 duration_frames 帧
        frames2 = []
        for i in range(duration_frames):
            ret, frame = cap2.read()
            if not ret:
                break
            frames2.append(frame)
        
        # 创建滑动转场帧
        for i in range(min(len(frames1), len(frames2))):
            progress = (i + 1) / len(frames1)
            
            # 创建空白画布
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            if direction == 'left':
                offset = int(width * progress)
                canvas[:, :width-offset] = frames1[i][:, offset:]
                canvas[:, width-offset:] = frames2[i][:, :offset]
            elif direction == 'right':
                offset = int(width * progress)
                canvas[:, offset:] = frames1[i][:, :width-offset]
                canvas[:, :offset] = frames2[i][:, width-offset:]
            elif direction == 'up':
                offset = int(height * progress)
                canvas[:height-offset, :] = frames1[i][offset:, :]
                canvas[height-offset:, :] = frames2[i][:offset, :]
            elif direction == 'down':
                offset = int(height * progress)
                canvas[offset:, :] = frames1[i][:height-offset, :]
                canvas[:offset, :] = frames2[i][height-offset:, :]
            
            out.write(canvas)
        
        # 写入第二个视频的剩余部分
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            out.write(frame)
        
        cap1.release()
        cap2.release()
        out.release()
        
        logger.info(f"✅ 滑动转场完成: {output_path}")
        return output_path
    
    @staticmethod
    def apply_transitions_to_videos(video_paths: List[str], output_dir: str,
                                   transition_type: str = 'fade',
                                   duration_frames: int = 15) -> List[str]:
        """
        对多个视频应用转场效果
        
        Args:
            video_paths: 视频路径列表
            output_dir: 输出目录
            transition_type: 转场类型 ('fade', 'cross_dissolve', 'slide')
            duration_frames: 转场持续帧数
            
        Returns:
            处理后的视频路径列表
        """
        import os
        
        if len(video_paths) < 2:
            return video_paths
        
        logger.info(f"对 {len(video_paths)} 个视频应用转场效果")
        
        result_paths = []
        
        for i in range(len(video_paths) - 1):
            output_path = os.path.join(
                output_dir, 
                f"transition_{i}_{i+1}.mp4"
            )
            
            if transition_type == 'fade':
                TransitionEffect.fade_transition(
                    video_paths[i], video_paths[i+1], 
                    output_path, duration_frames
                )
            elif transition_type == 'cross_dissolve':
                TransitionEffect.cross_dissolve(
                    video_paths[i], video_paths[i+1], 
                    output_path, duration_frames
                )
            elif transition_type.startswith('slide_'):
                direction = transition_type.split('_')[1]
                TransitionEffect.slide_transition(
                    video_paths[i], video_paths[i+1], 
                    output_path, direction, duration_frames
                )
            
            result_paths.append(output_path)
        
        logger.info(f"✅ 转场效果应用完成")
        return result_paths
