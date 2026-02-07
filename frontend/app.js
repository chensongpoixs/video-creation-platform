// 前端交互逻辑

let currentTaskId = null;
let pollInterval = null;

// 表单提交
document.getElementById('videoForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) return;
    
    // 禁用按钮
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    
    submitBtn.disabled = true;
    btnText.textContent = '提交中...';
    btnSpinner.classList.remove('d-none');
    
    try {
        const response = await fetch('/api/tasks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt })
        });
        
        if (!response.ok) {
            throw new Error('任务提交失败');
        }
        
        const data = await response.json();
        currentTaskId = data.task_id;
        
        // 显示任务状态
        showTaskStatus(data.task_id, data.status);
        
        // 开始轮询任务状态
        startPolling(data.task_id);
        
        // 清空输入框
        document.getElementById('prompt').value = '';
        
    } catch (error) {
        alert('错误: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = '生成视频';
        btnSpinner.classList.add('d-none');
    }
});

// 显示任务状态
function showTaskStatus(taskId, status) {
    const taskStatusDiv = document.getElementById('taskStatus');
    const taskIdSpan = document.getElementById('taskId');
    const statusSpan = document.getElementById('status');
    const progressBar = document.getElementById('progressBar');
    
    taskStatusDiv.classList.remove('d-none');
    taskIdSpan.textContent = taskId;
    
    updateStatus(status);
    
    if (status === 'pending' || status === 'processing') {
        progressBar.classList.remove('d-none');
    } else {
        progressBar.classList.add('d-none');
    }
}

// 更新状态显示
function updateStatus(status) {
    const statusSpan = document.getElementById('status');
    const statusMap = {
        'pending': { text: '等待中', class: 'bg-secondary' },
        'processing': { text: '生成中', class: 'bg-info' },
        'completed': { text: '已完成', class: 'bg-success' },
        'failed': { text: '失败', class: 'bg-danger' }
    };
    
    const statusInfo = statusMap[status] || { text: status, class: 'bg-secondary' };
    statusSpan.textContent = statusInfo.text;
    statusSpan.className = 'badge ' + statusInfo.class;
}

// 开始轮询任务状态
function startPolling(taskId) {
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/tasks/${taskId}`);
            if (!response.ok) {
                throw new Error('查询任务失败');
            }
            
            const data = await response.json();
            updateStatus(data.status);
            
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                showVideoPreview(data.result);
                loadTaskList();
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                alert('视频生成失败');
            }
        } catch (error) {
            console.error('轮询错误:', error);
        }
    }, 2000);
}

// 显示视频预览
function showVideoPreview(videoPath) {
    const videoPreviewDiv = document.getElementById('videoPreview');
    const videoPlayer = document.getElementById('videoPlayer');
    const downloadBtn = document.getElementById('downloadBtn');
    const progressBar = document.getElementById('progressBar');
    
    progressBar.classList.add('d-none');
    videoPreviewDiv.classList.remove('d-none');
    
    // 设置视频源
    videoPlayer.src = '/' + videoPath;
    downloadBtn.href = '/' + videoPath;
}

// 加载任务列表
async function loadTaskList() {
    try {
        const response = await fetch('/api/tasks');
        if (!response.ok) return;
        
        const tasks = await response.json();
        const taskListContent = document.getElementById('taskListContent');
        
        if (tasks.length === 0) {
            taskListContent.innerHTML = '<p class="text-muted">暂无任务记录</p>';
            return;
        }
        
        const html = tasks.map(task => `
            <div class="card mb-2">
                <div class="card-body py-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <small class="text-muted">${task.task_id.substring(0, 8)}...</small>
                            <p class="mb-0">${task.prompt.substring(0, 50)}...</p>
                        </div>
                        <span class="badge ${getStatusClass(task.status)}">${getStatusText(task.status)}</span>
                    </div>
                </div>
            </div>
        `).join('');
        
        taskListContent.innerHTML = html;
    } catch (error) {
        console.error('加载任务列表失败:', error);
    }
}

function getStatusClass(status) {
    const map = {
        'pending': 'bg-secondary',
        'processing': 'bg-info',
        'completed': 'bg-success',
        'failed': 'bg-danger'
    };
    return map[status] || 'bg-secondary';
}

function getStatusText(status) {
    const map = {
        'pending': '等待中',
        'processing': '生成中',
        'completed': '已完成',
        'failed': '失败'
    };
    return map[status] || status;
}

// 页面加载时获取任务列表
window.addEventListener('load', loadTaskList);
