"""
LLM 服务测试
"""
import pytest
from services.llm_service import (
    generate_script,
    parse_llm_response,
    validate_and_fix_script,
    generate_fallback_script
)

def test_generate_fallback_script():
    """测试备用脚本生成"""
    prompt = "制作一段关于森林探险的短视频，包含河流和小动物"
    script = generate_fallback_script(prompt)
    
    assert "scenes" in script
    assert len(script["scenes"]) >= 3
    assert script["scenes"][0]["scene_number"] == 1
    assert "description" in script["scenes"][0]
    assert "duration" in script["scenes"][0]

def test_parse_llm_response():
    """测试 JSON 解析"""
    response = '''
    {
      "title": "测试视频",
      "scenes": [
        {"scene_number": 1, "description": "场景1", "duration": 5}
      ]
    }
    '''
    script = parse_llm_response(response)
    assert script["title"] == "测试视频"
    assert len(script["scenes"]) == 1

def test_validate_and_fix_script():
    """测试脚本验证和修正"""
    script = {
        "scenes": [
            {"description": "场景1"},
            {"description": "场景2", "duration": 3}
        ]
    }
    
    fixed = validate_and_fix_script(script)
    
    assert "title" in fixed
    assert fixed["scenes"][0]["scene_number"] == 1
    assert fixed["scenes"][0]["duration"] == 5  # 默认值
    assert "total_duration" in fixed

def test_generate_script():
    """测试完整脚本生成"""
    prompt = "制作一段关于海滩日落的视频"
    script = generate_script(prompt)
    
    assert "scenes" in script
    assert len(script["scenes"]) > 0
    assert "title" in script
    assert "total_duration" in script
