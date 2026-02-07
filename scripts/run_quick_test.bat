@echo off
REM 快速测试脚本 - 只测试基础功能

echo ============================================================
echo 快速测试流程
echo ============================================================

echo.
echo 1. 环境验证...
python scripts\verify_setup.py

echo.
echo 2. 脚本生成测试...
python tests\test_script_generation.py

echo.
echo 3. 单场景视频生成测试...
python tests\test_single_scene.py

echo.
echo ============================================================
echo 快速测试完成！
echo ============================================================
pause
