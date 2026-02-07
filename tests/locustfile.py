"""
Locust 压力测试文件

运行方法:
1. 安装 Locust: pip install locust
2. 启动测试: locust -f tests/locustfile.py --host=http://localhost:8000
3. 打开浏览器: http://localhost:8089
4. 配置用户数和启动速率，开始测试
"""
from locust import HttpUser, task, between, events
import json
import random


class VideoUser(HttpUser):
    """视频平台用户行为模拟"""
    
    # 用户等待时间（秒）
    wait_time = between(1, 3)
    
    def on_start(self):
        """用户开始时执行（登录）"""
        # 注册用户
        username = f"user_{random.randint(1000, 9999)}"
        response = self.client.post("/api/auth/register", json={
            "username": username,
            "email": f"{username}@test.com",
            "password": "Test1234"
        }, catch_response=True)
        
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"注册失败: {response.text}")
            return
        
        # 登录
        response = self.client.post("/api/auth/login", json={
            "username": username,
            "password": "Test1234"
        }, catch_response=True)
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            response.success()
        else:
            response.failure(f"登录失败: {response.text}")
    
    @task(3)
    def get_health(self):
        """健康检查（高频）"""
        self.client.get("/health")
    
    @task(2)
    def get_model_status(self):
        """获取模型状态"""
        self.client.get("/api/model/status")
    
    @task(1)
    def get_user_info(self):
        """获取用户信息"""
        if hasattr(self, 'token'):
            self.client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {self.token}"}
            )
    
    @task(1)
    def create_task(self):
        """创建视频任务（低频但重要）"""
        if not hasattr(self, 'token'):
            return
        
        prompts = [
            "一个关于春天的短视频",
            "城市夜景延时摄影",
            "海边日落的美景",
            "森林中的小溪",
            "雪山风光"
        ]
        
        response = self.client.post(
            "/api/tasks",
            json={
                "prompt": random.choice(prompts),
                "num_scenes": random.randint(1, 3)
            },
            headers={"Authorization": f"Bearer {self.token}"},
            catch_response=True
        )
        
        if response.status_code == 200:
            data = response.json()
            self.task_id = data.get("task_id")
            response.success()
        else:
            response.failure(f"创建任务失败: {response.text}")
    
    @task(2)
    def get_task_status(self):
        """查询任务状态"""
        if not hasattr(self, 'token') or not hasattr(self, 'task_id'):
            return
        
        self.client.get(
            f"/api/tasks/{self.task_id}",
            headers={"Authorization": f"Bearer {self.token}"}
        )


class AdminUser(HttpUser):
    """管理员用户行为模拟"""
    
    wait_time = between(5, 10)
    
    @task
    def get_system_metrics(self):
        """获取系统指标"""
        self.client.get("/api/metrics")
    
    @task
    def get_performance_report(self):
        """获取性能报告"""
        self.client.get("/api/performance/report")


class APIStressTest(HttpUser):
    """API 压力测试"""
    
    wait_time = between(0.1, 0.5)  # 高频请求
    
    @task(10)
    def rapid_health_check(self):
        """快速健康检查"""
        self.client.get("/health")
    
    @task(5)
    def rapid_model_status(self):
        """快速模型状态查询"""
        self.client.get("/api/model/status")


# 事件监听器
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时"""
    print("=" * 60)
    print("压力测试开始")
    print(f"目标主机: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束时"""
    print("=" * 60)
    print("压力测试结束")
    print("=" * 60)
    
    # 打印统计信息
    stats = environment.stats
    print(f"\n总请求数: {stats.total.num_requests}")
    print(f"失败请求数: {stats.total.num_failures}")
    print(f"平均响应时间: {stats.total.avg_response_time:.2f}ms")
    print(f"最大响应时间: {stats.total.max_response_time:.2f}ms")
    print(f"请求速率: {stats.total.total_rps:.2f} RPS")
    
    if stats.total.num_requests > 0:
        failure_rate = stats.total.num_failures / stats.total.num_requests * 100
        print(f"失败率: {failure_rate:.2f}%")


# 自定义场景
class QuickTest(HttpUser):
    """快速测试场景（用于快速验证）"""
    
    wait_time = between(1, 2)
    
    @task
    def quick_test(self):
        """快速测试"""
        # 健康检查
        self.client.get("/health")
        
        # 模型状态
        self.client.get("/api/model/status")


# 使用说明
"""
运行不同的测试场景:

1. 基础压力测试（模拟真实用户）:
   locust -f tests/locustfile.py --host=http://localhost:8000 --users=10 --spawn-rate=2

2. 高并发压力测试:
   locust -f tests/locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10

3. API 压力测试（高频请求）:
   locust -f tests/locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=10 APIStressTest

4. 快速测试:
   locust -f tests/locustfile.py --host=http://localhost:8000 --users=5 --spawn-rate=1 QuickTest --headless --run-time=1m

5. 无界面模式（自动化测试）:
   locust -f tests/locustfile.py --host=http://localhost:8000 --users=20 --spawn-rate=5 --headless --run-time=5m --html=report.html

参数说明:
--users: 模拟用户数
--spawn-rate: 每秒启动的用户数
--run-time: 运行时间（如 1m, 5m, 1h）
--headless: 无界面模式
--html: 生成 HTML 报告
"""
