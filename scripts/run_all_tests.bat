@echo off
REM 完整测试流程脚本 (Windows)

echo ============================================================
echo 开始完整测试流程
echo ============================================================

REM 1. 环境验证
echo.
echo 1. 环境验证...
echo ------------------------------------------------------------
python scripts\verify_setup.py
if %errorlevel% neq 0 (
    echo 环境验证失败，请检查配置
    pause
    exit /b 1
)

REM 2. 单元测试
echo.
echo 2. 单元测试...
echo ------------------------------------------------------------
pytest tests\test_llm_service.py tests\test_video_service.py -v
if %errorlevel% neq 0 (
    echo 单元测试失败
)

REM 3. 模型加载测试
echo.
echo 3. 模型加载测试...
echo ------------------------------------------------------------
python tests\test_model_loading.py
if %errorlevel% neq 0 (
    echo 模型加载测试失败
)

REM 4. 脚本生成测试
echo.
echo 4. 脚本生成测试...
echo ------------------------------------------------------------
python tests\test_script_generation.py
if %errorlevel% neq 0 (
    echo 脚本生成测试失败
)

REM 5. 单场景视频生成测试
echo.
echo 5. 单场景视频生成测试...
echo ------------------------------------------------------------
python tests\test_single_scene.py
if %errorlevel% neq 0 (
    echo 单场景视频生成测试失败
)

REM 6. 端到端测试
echo.
echo 6. 端到端测试...
echo ------------------------------------------------------------
python tests\test_end_to_end.py
if %errorlevel% neq 0 (
    echo 端到端测试失败
)

REM 7. 性能测试
echo.
echo 7. 性能测试...
echo ------------------------------------------------------------
python tests\test_benchmark.py

echo.
echo ============================================================
echo 测试完成！
echo ============================================================
pause
