#!/bin/bash

# Router-R1 单卡简化训练脚本
# 适用于单张3090显卡（24GB显存）

export CUDA_VISIBLE_DEVICES=0  # 只使用第一张卡
export DATA_DIR='data/mock_data'  # 使用模拟数据

WAND_PROJECT='Router-R1-Single-GPU-Demo'

# 使用较小的模型以适应单卡环境
export BASE_MODEL='microsoft/DialoGPT-small'  # 使用轻量级模型
export EXPERIMENT_NAME=single-gpu-demo-$(date +%Y%m%d-%H%M%S)

echo "🚀 启动Router-R1单卡演示训练"
echo "📊 实验名称: $EXPERIMENT_NAME"
echo "🎯 基础模型: $BASE_MODEL"
echo "💾 数据目录: $DATA_DIR"

# 设置环境变量以优化内存使用
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 创建模拟数据目录
mkdir -p $DATA_DIR

# 如果没有数据文件，创建一个简单的模拟数据文件
if [ ! -f "$DATA_DIR/train_demo.parquet" ]; then
    echo "📝 创建模拟训练数据..."
    python3 -c "
import pandas as pd
import os

# 创建模拟数据
data = {
    'question': [
        '什么是人工智能？',
        '机器学习的基本原理是什么？',
        '深度学习有哪些应用？',
        '自然语言处理的发展前景如何？',
        '计算机视觉技术的核心是什么？'
    ],
    'answer': [
        '人工智能是计算机科学的一个分支...',
        '机器学习通过算法让计算机从数据中学习...',
        '深度学习在图像识别、自然语言处理等领域有广泛应用...',
        '自然语言处理将继续发展，实现更好的人机交互...',
        '计算机视觉的核心是图像识别和理解...'
    ]
}

df = pd.DataFrame(data)
os.makedirs('$DATA_DIR', exist_ok=True)
df.to_parquet('$DATA_DIR/train_demo.parquet', index=False)
df.to_parquet('$DATA_DIR/test_demo.parquet', index=False)
print('✅ 模拟数据创建完成')
"
fi

echo "⚙️ 开始简化训练流程..."

# 注意：这里只是演示配置，实际的verl训练需要更复杂的设置
# 由于系统盘空间限制，我们只运行一个验证流程

python3 -c "
print('🎯 Router-R1 简化训练演示')
print('=' * 50)
print('📋 训练配置:')
print(f'  - 基础模型: $BASE_MODEL')
print(f'  - 实验名称: $EXPERIMENT_NAME')
print(f'  - GPU设备: $CUDA_VISIBLE_DEVICES')
print(f'  - 数据目录: $DATA_DIR')
print('')
print('💡 在真实环境中，这里会启动完整的PPO训练流程')
print('   包括Actor、Critic、Rollout等组件的训练')
print('')
print('✨ 由于系统资源限制，此演示展示了配置过程')
print('   要运行完整训练，请使用提供的 train.sh 脚本')
print('   并确保有足够的GPU内存和存储空间')
"

echo ""
echo "🎉 简化训练演示完成！"
echo ""
echo "📚 接下来的步骤："
echo "1. 运行简化推理演示: python simple_demo.py"
echo "2. 如需完整训练，请配置API并使用原始 train.sh"
echo "3. 检查生成的配置和数据文件"