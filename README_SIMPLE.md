# Router-R1 单卡简化运行指南

## 🎯 项目简介

Router-R1 是一个用于教学大型语言模型进行多轮路由和聚合的系统。这个简化版本专门为单张 3090 显卡（24GB 显存）和有限系统盘空间（30GB）优化。

## 🚀 快速开始

### 环境要求
- 单张 RTX 3090 (24GB 显存)
- Python 3.9+
- PyTorch 2.0+
- 系统盘可用空间 > 5GB

### 已安装的核心依赖
```bash
transformers==4.47.0
datasets
accelerate
pandas
```

## 📋 运行演示

### 1. 简化推理演示
```bash
# 基本演示
python simple_demo.py

# 自定义问题
python simple_demo.py --question "你的问题"
```

### 2. 训练流程演示
```bash
# 运行简化训练演示
bash single_gpu_train.sh
```

## 🔧 功能说明

### 简化推理演示 (`simple_demo.py`)
- ✅ 展示多轮路由推理流程
- ✅ 模拟不同 LLM 的响应
- ✅ 演示 `<think>`, `<search>`, `<answer>` 标签的使用
- ✅ 无需真实 API 调用

### 简化训练演示 (`single_gpu_train.sh`)
- ✅ 创建模拟训练数据
- ✅ 展示单卡训练配置
- ✅ 优化内存使用参数
- ✅ 适合 24GB 显存限制

## 🎮 演示输出示例

```
🌟 欢迎使用 Router-R1 简化演示！
🎯 这个演示展示了多LLM路由和聚合的核心概念

🔄 第 1 轮推理:
<think>
我需要了解更多关于问题的信息...
</think>
<search>Qwen2.5-7B-Instruct:详细查询</search>

🔍 检测到查询: Qwen2.5-7B-Instruct:详细查询
📥 路由结果: 根据Qwen2.5-7B的知识库...

✅ 检测到最终答案，推理完成！
```

## 📁 项目结构

```
Router-R1/
├── simple_demo.py          # 简化推理演示
├── single_gpu_train.sh     # 单卡训练演示
├── data_process/           # 数据处理模块
│   └── prompt_pool.py      # 提示词模板
├── router_r1/llm_agent/    # 路由服务
│   └── route_service.py    # 核心路由逻辑
├── verl/                   # 强化学习框架
└── data/mock_data/         # 模拟数据（自动生成）
```

## ⚙️ 资源优化

### 内存优化
- 使用梯度检查点降低显存占用
- 启用参数/梯度/优化器卸载
- 设置合适的批次大小

### 存储优化
- 不缓存下载的模型文件 (`--no-cache-dir`)
- 使用轻量级模型进行演示
- 自动清理临时文件

## 🔧 进阶配置

如果您想运行完整版本：

1. **配置 API 密钥**：
   ```bash
   # 编辑 train.sh 或 test.sh
   export api_base="YOUR_API_BASE"
   export api_key="YOUR_API_KEY"
   ```

2. **准备真实数据**：
   ```bash
   # 创建数据目录
   mkdir -p data/nq_search
   # 放置训练和测试数据文件
   ```

3. **运行完整训练**：
   ```bash
   # 使用原始训练脚本（需要更多资源）
   bash train.sh
   ```

## 🎯 核心概念

### 多轮路由
- 模型可以调用不同的专业 LLM
- 每个 LLM 有特定的专长领域
- 通过多轮交互获得更好的答案

### 路由策略
- 根据问题类型选择合适的模型
- 考虑模型成本和性能平衡
- 支持并行和串行调用

### 强化学习训练
- 使用 PPO 算法优化路由策略
- Actor-Critic 架构
- 基于奖励信号学习最优路由

## 🚨 注意事项

1. **内存限制**：简化版本已优化单卡使用
2. **存储空间**：项目占用约 6MB，依赖包约 1GB
3. **API 调用**：演示版本使用模拟响应，无需真实 API
4. **模型下载**：避免下载大型模型以节省空间

## 🔍 故障排除

### 常见问题

1. **GPU 内存不足**
   ```bash
   # 使用 CPU 模式
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **磁盘空间不足**
   ```bash
   # 清理 pip 缓存
   pip cache purge
   ```

3. **依赖冲突**
   ```bash
   # 重新安装核心依赖
   pip install transformers==4.47.0 --no-cache-dir
   ```

## 📚 学习资源

- 查看 `data_process/prompt_pool.py` 了解提示词设计
- 阅读 `router_r1/llm_agent/route_service.py` 理解路由逻辑
- 探索 `verl/` 目录学习强化学习框架

## 🎉 下一步

1. 运行所有演示脚本
2. 修改问题测试不同场景
3. 探索项目源码结构
4. 考虑扩展到多卡环境（如果有更多资源）

---

🚀 **开始体验 Router-R1 的智能路由系统吧！**