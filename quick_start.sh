#!/bin/bash

echo "🌟 Router-R1 一键演示启动器"
echo "=============================================="

# 检查环境
echo "🔧 检查运行环境..."
python -c "
import torch
print(f'✅ Python版本: {__import__(\"sys\").version.split()[0]}')
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
else:
    print('⚠️  GPU不可用，将使用CPU模式')
"

echo ""
echo "📊 检查磁盘空间..."
df -h / | grep overlay

echo ""
echo "🚀 可用的演示选项："
echo ""
echo "1️⃣  推理演示 - 展示多轮路由推理"
echo "2️⃣  训练演示 - 展示训练配置"
echo "3️⃣  自定义问题演示"
echo "4️⃣  查看项目结构"
echo "5️⃣  清理和退出"

echo ""
read -p "请选择要运行的演示 (1-5): " choice

case $choice in
    1)
        echo "🎯 启动推理演示..."
        python simple_demo.py --question "什么是Router-R1系统的核心优势"
        ;;
    2)
        echo "🎯 启动训练演示..."
        bash single_gpu_train.sh
        ;;
    3)
        echo "🎯 自定义问题演示..."
        read -p "请输入您的问题: " user_question
        python simple_demo.py --question "$user_question"
        ;;
    4)
        echo "📁 项目结构:"
        tree -L 3 -I "__pycache__|*.pyc|*.egg-info"
        echo ""
        echo "📚 主要文件说明:"
        echo "  - simple_demo.py: 简化推理演示"
        echo "  - single_gpu_train.sh: 单卡训练演示"
        echo "  - README_SIMPLE.md: 详细使用说明"
        echo "  - data_process/: 数据处理和提示词模板"
        echo "  - router_r1/: 核心路由服务逻辑"
        echo "  - verl/: 强化学习训练框架"
        ;;
    5)
        echo "🧹 清理临时文件..."
        rm -rf data/mock_data
        echo "✅ 清理完成"
        echo "👋 感谢使用 Router-R1 演示系统！"
        ;;
    *)
        echo "❌ 无效选择，请重新运行脚本"
        ;;
esac

echo ""
echo "=============================================="
echo "💡 提示: 更多详细信息请查看 README_SIMPLE.md"
echo "🔧 如需完整功能，请配置API并使用原始脚本"