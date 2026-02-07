"""
LLM服务模块 - 负责脚本生成和分镜拆解
"""
import json
import re
from typing import Dict, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 提示词模板
SCRIPT_GENERATION_PROMPT = """你是一个专业的视频脚本创作助手。请根据用户的创作指令，生成详细的视频脚本和分镜。

用户指令：{user_prompt}

请按照以下 JSON 格式输出：
{{
  "title": "视频标题",
  "total_duration": 总时长（秒）,
  "scenes": [
    {{
      "scene_number": 1,
      "description": "场景描述（详细的视觉描述，包含环境、人物、动作等）",
      "duration": 5,
      "camera": "镜头类型（wide shot/close up/medium shot/aerial view）",
      "action": "动作描述"
    }}
  ]
}}

要求：
1. 每个场景描述要具体、生动，便于视频生成
2. 场景之间要有连贯性和故事性
3. 每个场景时长建议 3-8 秒
4. 至少生成 3 个场景，最多 8 个场景
5. 只输出 JSON 格式，不要其他内容
6. 确保 JSON 格式正确，可以被解析
"""

def generate_script(prompt: str) -> Dict:
    """
    使用 LLM 生成视频脚本
    
    Args:
        prompt: 用户输入的创作指令
        
    Returns:
        包含分镜信息的字典
    """
    try:
        logger.info(f"开始生成脚本，用户输入: {prompt}")
        
        # 尝试使用 LLM 生成
        try:
            from services.model_loader import llm_loader
            
            if llm_loader.is_loaded:
                # 构造完整提示词
                full_prompt = SCRIPT_GENERATION_PROMPT.format(user_prompt=prompt)
                
                # 调用 LLM 生成
                response = llm_loader.generate(
                    full_prompt,
                    max_length=2048,
                    temperature=0.7
                )
                
                logger.info(f"LLM 原始输出: {response[:200]}...")
                
                # 解析 JSON
                script = parse_llm_response(response)
                
                # 验证和修正
                script = validate_and_fix_script(script)
                
                logger.info(f"✅ 脚本生成成功（LLM），共 {len(script['scenes'])} 个场景")
                return script
            else:
                logger.warning("LLM 模型未加载，使用备用方案")
                
        except Exception as e:
            logger.warning(f"LLM 生成失败: {str(e)}，使用备用方案")
        
        # 备用方案：简单分句
        return generate_fallback_script(prompt)
        
    except Exception as e:
        logger.error(f"脚本生成失败: {str(e)}")
        return generate_fallback_script(prompt)

def parse_llm_response(response: str) -> Dict:
    """解析 LLM 输出的 JSON"""
    try:
        # 方法1: 直接解析
        try:
            return json.loads(response)
        except:
            pass
        
        # 方法2: 提取 JSON 部分
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        
        # 方法3: 提取代码块中的 JSON
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
            return json.loads(json_str)
        
        raise ValueError("未找到有效的 JSON 格式")
        
    except Exception as e:
        logger.error(f"JSON 解析失败: {str(e)}")
        raise

def validate_and_fix_script(script: Dict) -> Dict:
    """验证和修正脚本格式"""
    # 确保必要字段存在
    if "title" not in script:
        script["title"] = "自动生成视频"
    
    if "scenes" not in script or not script["scenes"]:
        raise ValueError("脚本中没有场景")
    
    # 限制场景数量
    if len(script["scenes"]) > 8:
        logger.warning(f"场景数量过多({len(script['scenes'])}), 截取前8个")
        script["scenes"] = script["scenes"][:8]
    
    # 修正场景编号和字段
    for i, scene in enumerate(script["scenes"]):
        scene["scene_number"] = i + 1
        
        # 确保必要字段
        if "description" not in scene or not scene["description"]:
            scene["description"] = f"场景 {i+1}"
        
        if "duration" not in scene or not isinstance(scene["duration"], (int, float)):
            scene["duration"] = 5
        else:
            # 限制时长范围
            scene["duration"] = max(3, min(10, scene["duration"]))
        
        if "camera" not in scene:
            scene["camera"] = "wide shot"
        
        if "action" not in scene:
            scene["action"] = "展示场景"
    
    # 计算总时长
    script["total_duration"] = sum(s["duration"] for s in script["scenes"])
    
    return script

def generate_fallback_script(prompt: str) -> Dict:
    """生成备用脚本（当 LLM 失败时）"""
    logger.info("使用备用脚本生成方案")
    
    # 智能分句
    sentences = re.split(r'[，。,.]', prompt)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
    
    # 如果分句太少，使用整个提示词
    if len(sentences) < 2:
        sentences = [prompt]
    
    # 生成场景
    scenes = []
    camera_types = ["wide shot", "medium shot", "close up", "aerial view"]
    
    for i, sentence in enumerate(sentences[:6]):  # 最多6个场景
        scenes.append({
            "scene_number": i + 1,
            "description": sentence,
            "duration": 5,
            "camera": camera_types[i % len(camera_types)],
            "action": "展示场景内容"
        })
    
    # 确保至少有3个场景
    while len(scenes) < 3:
        scenes.append({
            "scene_number": len(scenes) + 1,
            "description": f"{prompt} - 场景 {len(scenes) + 1}",
            "duration": 5,
            "camera": "wide shot",
            "action": "展示场景"
        })
    
    script = {
        "title": "自动生成视频",
        "total_duration": len(scenes) * 5,
        "scenes": scenes
    }
    
    logger.info(f"✅ 备用脚本生成完成，共 {len(scenes)} 个场景")
    return script

def optimize_prompt_for_video(scene_description: str) -> str:
    """
    优化场景描述为视频生成模型的 Prompt
    
    Args:
        scene_description: 场景描述
        
    Returns:
        优化后的 Prompt
    """
    # 添加视觉质量关键词
    quality_keywords = "high quality, cinematic, detailed, 4k, professional lighting"
    
    # 构造完整 Prompt
    prompt = f"{scene_description}, {quality_keywords}"
    
    return prompt
