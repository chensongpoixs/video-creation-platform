"""
字幕系统模块
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SubtitleSystem:
    """字幕系统类"""
    
    def __init__(self, font_path: str = None, font_size: int = 32,
                 font_color: Tuple[int, int, int] = (255, 255, 255),
                 bg_color: Tuple[int, int, int, int] = None,
                 position: str = 'bottom'):
        """
        初始化字幕系统
        
        Args:
            font_path: 字体文件路径（None 使用默认字体）
            font_size: 字体大小
            font_color: 字体颜色 (R, G, B)
            bg_color: 背景颜色 (R, G, B, A)，None 为透明
            position: 字幕位置 ('top', 'center', 'bottom')
        """
        self.font_path = font_path
        self.font_size = font_size
        self.font_color = font_color
        self.bg_color = bg_color
        self.position = position
        
        # 加载字体
        try:
            if font_path:
                self.font = ImageFont.truetype(font_path, font_size)
            else:
                # 尝试使用系统默认字体
                self.font = ImageFont.load_default()
                logger.warning("使用默认字体，可能不支持中文")
        except Exception as e:
            logger.warning(f"字体加载失败: {e}，使用默认字体")
            self.font = ImageFont.load_default()
    
    def add_subtitle_to_frame(self, frame: np.ndarray, text: str,
                             position: Tuple[int, int] = None) -> np.ndarray:
        """
        在单帧上添加字幕
        
        Args:
            frame: 输入帧（BGR格式）
            text: 字幕文本
            position: 自定义位置 (x, y)，None 使用默认位置
            
        Returns:
            添加字幕后的帧
        """
        if not text:
            return frame
        
        # 转换为 PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        
        # 获取文本大小
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算位置
        if position is None:
            img_width, img_height = pil_img.size
            
            if self.position == 'top':
                x = (img_width - text_width) // 2
                y = 30
            elif self.position == 'center':
                x = (img_width - text_width) // 2
                y = (img_height - text_height) // 2
            else:  # bottom
                x = (img_width - text_width) // 2
                y = img_height - text_height - 50
            
            position = (x, y)
        
        # 绘制背景（如果有）
        if self.bg_color:
            padding = 10
            bg_bbox = [
                position[0] - padding,
                position[1] - padding,
                position[0] + text_width + padding,
                position[1] + text_height + padding
            ]
            draw.rectangle(bg_bbox, fill=self.bg_color)
        
        # 绘制文字阴影（增强可读性）
        shadow_offset = 2
        draw.text(
            (position[0] + shadow_offset, position[1] + shadow_offset),
            text, font=self.font, fill=(0, 0, 0, 180)
        )
        
        # 绘制文字
        draw.text(position, text, font=self.font, fill=self.font_color)
        
        # 转换回 OpenCV 格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def add_subtitles_to_video(self, input_path: str, output_path: str,
                               subtitles: List[Dict]) -> str:
        """
        为视频添加字幕
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            subtitles: 字幕列表，格式:
                [
                    {"text": "字幕文本", "start": 0.0, "end": 5.0},
                    {"text": "下一句", "start": 5.0, "end": 10.0},
                ]
            
        Returns:
            输出视频路径
        """
        logger.info(f"开始添加字幕: {input_path}")
        logger.info(f"字幕数量: {len(subtitles)}")
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 将字幕时间转换为帧号
        subtitle_frames = []
        for sub in subtitles:
            start_frame = int(sub['start'] * fps)
            end_frame = int(sub['end'] * fps)
            subtitle_frames.append({
                'text': sub['text'],
                'start_frame': start_frame,
                'end_frame': end_frame
            })
        
        # 处理每一帧
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 查找当前帧对应的字幕
            current_subtitle = None
            for sub in subtitle_frames:
                if sub['start_frame'] <= frame_count < sub['end_frame']:
                    current_subtitle = sub['text']
                    break
            
            # 添加字幕
            if current_subtitle:
                frame = self.add_subtitle_to_frame(frame, current_subtitle)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.debug(f"已处理 {frame_count}/{total_frames} 帧")
        
        cap.release()
        out.release()
        
        logger.info(f"✅ 字幕添加完成: {output_path}")
        return output_path
    
    @staticmethod
    def generate_subtitles_from_script(script: Dict) -> List[Dict]:
        """
        从脚本生成字幕
        
        Args:
            script: 脚本字典
            
        Returns:
            字幕列表
        """
        subtitles = []
        current_time = 0.0
        
        for scene in script.get('scenes', []):
            duration = scene.get('duration', 5)
            description = scene.get('description', '')
            
            # 简化描述作为字幕
            subtitle_text = description[:50] + '...' if len(description) > 50 else description
            
            subtitles.append({
                'text': subtitle_text,
                'start': current_time,
                'end': current_time + duration
            })
            
            current_time += duration
        
        return subtitles
    
    @staticmethod
    def parse_srt_file(srt_path: str) -> List[Dict]:
        """
        解析 SRT 字幕文件
        
        Args:
            srt_path: SRT 文件路径
            
        Returns:
            字幕列表
        """
        subtitles = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的 SRT 解析
            blocks = content.strip().split('\n\n')
            
            for block in blocks:
                lines = block.split('\n')
                if len(lines) >= 3:
                    # 解析时间
                    time_line = lines[1]
                    times = time_line.split(' --> ')
                    
                    if len(times) == 2:
                        start = SubtitleSystem._parse_srt_time(times[0])
                        end = SubtitleSystem._parse_srt_time(times[1])
                        text = '\n'.join(lines[2:])
                        
                        subtitles.append({
                            'text': text,
                            'start': start,
                            'end': end
                        })
        
        except Exception as e:
            logger.error(f"SRT 文件解析失败: {e}")
        
        return subtitles
    
    @staticmethod
    def _parse_srt_time(time_str: str) -> float:
        """
        解析 SRT 时间格式 (HH:MM:SS,mmm)
        
        Args:
            time_str: 时间字符串
            
        Returns:
            秒数
        """
        time_str = time_str.strip().replace(',', '.')
        parts = time_str.split(':')
        
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        
        return hours * 3600 + minutes * 60 + seconds
