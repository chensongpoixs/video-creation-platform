"""
API测试用例
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# 添加backend目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app

client = TestClient(app)

def test_health_check():
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_create_task():
    """测试创建任务"""
    response = client.post(
        "/api/tasks/",
        json={"prompt": "制作一段关于森林探险的短视频"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"

def test_get_task():
    """测试获取任务详情"""
    # 先创建任务
    create_response = client.post(
        "/api/tasks/",
        json={"prompt": "测试视频"}
    )
    task_id = create_response.json()["task_id"]
    
    # 获取任务详情
    response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id

def test_list_tasks():
    """测试获取任务列表"""
    response = client.get("/api/tasks/")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert "total" in data
