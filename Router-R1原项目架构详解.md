# Router-R1 原项目架构详解

## 🎯 项目概述

**Router-R1** 是一个基于强化学习的大型语言模型多轮路由和聚合系统，由 U Lab (UIUC) 开发。该系统将多模型路由问题转化为序列决策过程，实现了智能的多轮推理和模型协作。

## 🏗️ 核心架构

### 系统设计理念

Router-R1 摒弃了传统的单轮路由（一对一映射）模式，采用**多轮路由**架构：

```
传统单轮路由: Query → 单个模型 → Response
Router-R1多轮: Query → Think → Route → Integrate → Route → ... → Answer
```

### 架构组件

```
Router-R1/
├── 🧠 路由决策引擎 (router_r1/)
│   ├── route_service.py     # 核心路由服务
│   ├── generation.py        # 推理生成逻辑
│   └── tensor_helper.py     # 张量操作工具
├── 🚀 强化学习框架 (verl/)
│   ├── trainer/             # PPO训练器
│   ├── workers/             # 分布式工作器
│   ├── models/              # 模型定义
│   └── utils/               # 工具函数
├── 📊 数据处理 (data_process/)
│   ├── prompt_pool.py       # 提示词模板
│   ├── qa_train_merge.py    # 训练数据生成
│   ├── qa_test_merge.py     # 测试数据生成
│   └── qa_test_gen.py       # 数据生成工具
└── 🔧 配置脚本
    ├── train.sh             # 训练启动脚本
    ├── test.sh              # 测试脚本
    └── infer_vllm.py        # 推理脚本
```

## 🧠 核心技术架构

### 1. 多轮路由决策引擎

**路由服务核心** ([`route_service.py`](./router_r1/llm_agent/route_service.py))

```python
# 核心路由函数
def access_routing_pool(queries, api_base, api_key):
    """
    路由池管理 - 处理多查询并发路由
    支持模型并发调用，优化响应时间
    """

def check_llm_name(target_llm):
    """
    智能模型映射 - 将友好名称映射到API模型ID
    支持多厂商模型：OpenAI, Meta, Mistral, Google等
    """

def get_llm_response_via_api(prompt, LLM_MODEL, ...):
    """
    API调用封装 - 统一的模型调用接口
    包含重试机制、超时处理、错误恢复
    """
```

**支持的模型生态**:
- **Qwen系列**: qwen/qwen2.5-7b-instruct
- **LLaMA系列**: meta/llama-3.1-70b-instruct, meta/llama-3.1-8b-instruct
- **Mistral系列**: mistralai/mistral-7b-instruct-v0.3, mistralai/mixtral-8x22b-instruct
- **Gemma系列**: google/gemma-2-27b-it
- **专业模型**: nvidia/llama3-chatqa-1.5-8b, writer/palmyra-creative-122b

### 2. 多轮推理流程

**推理模式**:
```
<think>
内部推理 - 分析问题，确定需要的外部信息
</think>

<search>模型名:具体查询</search>
外部路由 - 调用专业模型获取信息

<information>
专家响应 - 集成外部模型的回答
</information>

<answer>
最终答案 - 综合所有信息给出结果
</answer>
```

**提示词模板** ([`prompt_pool.py`](./data_process/prompt_pool.py)):
- **PROMPT_TEMPLATE_QWEN**: 针对Qwen模型优化的4500+字符模板
- **PROMPT_TEMPLATE_LLAMA**: 针对LLaMA模型优化的推理模板
- 包含详细的模型描述和使用指南

### 3. 强化学习训练框架 (VERL)

**训练架构**:
```
VERL Framework
├── Actor (策略网络)
│   ├── 路由决策生成
│   ├── 模型选择策略
│   └── 多轮交互控制
├── Critic (价值网络)
│   ├── 状态价值估计
│   ├── 奖励信号评估
│   └── 策略优化指导
└── Rollout (推理引擎)
    ├── vLLM集成
    ├── 分布式推理
    └── 批处理优化
```

**核心训练组件** ([`verl/`](./verl/)):
- **[`trainer/main_ppo.py`](./verl/trainer/main_ppo.py)**: PPO算法实现
- **[`workers/`](./verl/workers/)**: Actor, Critic, Rollout工作器
- **[`models/`](./verl/models/)**: 支持LLaMA, Qwen等模型
- **[`utils/`](./verl/utils/)**: 分布式训练、日志、评估工具

### 4. 奖励机制设计

**多层次奖励函数**:
```python
total_reward = format_reward + outcome_reward + cost_reward

# 格式奖励: 确保输出符合<think><search><answer>格式
# 结果奖励: 基于Exact Match或F1-Score评估最终答案质量  
# 成本奖励: 平衡性能与API调用成本，支持cost_coe参数调节
```

**成本计算** ([`API_PRICE_1M_TOKENS`](./router_r1/llm_agent/route_service.py)):
```python
API_PRICE_1M_TOKENS = {
    "qwen/qwen2.5-7b-instruct": 0.3,
    "meta/llama-3.1-70b-instruct": 0.88,
    "meta/llama-3.1-8b-instruct": 0.18,
    # ... 更多模型定价
}
```

## 🚀 技术特性

### 分布式训练支持
- **多GPU训练**: 支持tensor并行和数据并行
- **FSDP优化**: 参数、梯度、优化器卸载
- **vLLM集成**: 高效的推理加速
- **Ray分布式**: 支持多节点扩展

### 模型适配性
- **Transformer架构**: 支持主流开源模型
- **API统一接口**: OpenAI API兼容
- **动态batch**: 优化GPU利用率
- **梯度检查点**: 降低显存占用

### 评估指标
- **Exact Match (EM)**: 精确匹配评估
- **F1-Score**: 语义相似度评估
- **Cost Efficiency**: 成本效益分析
- **Response Quality**: 响应质量评估

## 📊 数据流架构

### 训练数据流
```
原始QA数据集 → 数据预处理 → 多轮对话构造 → 训练样本生成
     ↓              ↓              ↓              ↓
NQ, HotpotQA → qa_train_merge → 路由标注 → PPO训练数据
```

### 推理数据流  
```
用户查询 → 初始提示词 → 多轮路由 → 结果聚合 → 最终答案
    ↓         ↓          ↓         ↓         ↓
Question → Template → Route → Integrate → Answer
```

## 🎯 核心优势

### 1. 智能路由策略
- **任务自适应**: 根据问题类型自动选择最优模型组合
- **成本优化**: 平衡性能和API调用成本
- **并发处理**: 支持多查询并行路由

### 2. 多轮协作机制
- **信息累积**: 每轮推理结果作为下轮输入
- **专家协作**: 不同模型贡献各自专长
- **动态终止**: 智能判断何时给出最终答案

### 3. 强化学习优化
- **策略学习**: 通过RL学习最优路由策略
- **奖励塑造**: 多维度奖励函数指导训练
- **在线优化**: 支持持续学习和策略改进

## 🔧 配置与扩展

### 自定义模型池
1. **模型描述配置**: 在[`prompt_pool.py`](./data_process/prompt_pool.py)中添加新模型描述
2. **路由映射更新**: 修改[`check_llm_name`](./router_r1/llm_agent/route_service.py)函数
3. **定价配置**: 更新[`API_PRICE_1M_TOKENS`](./router_r1/llm_agent/route_service.py)字典
4. **API集成**: 配置相应的API端点和密钥

### 训练参数调优
```bash
# 性能vs成本权衡
cost_coe=0.9  # 高成本敏感度

# 奖励指标选择  
reward_metric="em"    # 精确匹配
reward_metric="f1"    # F1分数

# 批处理优化
train_batch_size=64   # 建议使用大批次以提升稳定性
```

## 📚 相关研究

Router-R1基于以下研究基础：
- **强化学习**: PPO算法优化路由策略
- **多智能体系统**: 模型协作机制设计  
- **课程学习**: 渐进式训练策略
- **成本效益优化**: 性能-成本平衡算法

## 🏆 应用场景

- **复杂问答**: 多跳推理、知识整合
- **专业咨询**: 法律、医疗、技术领域
- **创意生成**: 多角度创意协作
- **代码开发**: 多语言编程支持
- **学术研究**: 文献综述、论文写作

---

**总结**: Router-R1 通过创新的多轮路由架构和强化学习优化，实现了大型语言模型的智能协作，在复杂任务上显著超越单模型性能，同时优化了成本效益。这一架构为未来的多模型系统设计提供了重要参考。# Router-R1 原项目架构详解

## 🎯 项目概述

**Router-R1** 是一个基于强化学习的大型语言模型多轮路由和聚合系统，由 U Lab (UIUC) 开发。该系统将多模型路由问题转化为序列决策过程，实现了智能的多轮推理和模型协作。

## 🏗️ 核心架构

### 系统设计理念

Router-R1 摒弃了传统的单轮路由（一对一映射）模式，采用**多轮路由**架构：

```
传统单轮路由: Query → 单个模型 → Response
Router-R1多轮: Query → Think → Route → Integrate → Route → ... → Answer
```

### 架构组件

```
Router-R1/
├── 🧠 路由决策引擎 (router_r1/)
│   ├── route_service.py     # 核心路由服务
│   ├── generation.py        # 推理生成逻辑
│   └── tensor_helper.py     # 张量操作工具
├── 🚀 强化学习框架 (verl/)
│   ├── trainer/             # PPO训练器
│   ├── workers/             # 分布式工作器
│   ├── models/              # 模型定义
│   └── utils/               # 工具函数
├── 📊 数据处理 (data_process/)
│   ├── prompt_pool.py       # 提示词模板
│   ├── qa_train_merge.py    # 训练数据生成
│   ├── qa_test_merge.py     # 测试数据生成
│   └── qa_test_gen.py       # 数据生成工具
└── 🔧 配置脚本
    ├── train.sh             # 训练启动脚本
    ├── test.sh              # 测试脚本
    └── infer_vllm.py        # 推理脚本
```

## 🧠 核心技术架构

### 1. 多轮路由决策引擎

**路由服务核心** ([`route_service.py`](./router_r1/llm_agent/route_service.py))

```python
# 核心路由函数
def access_routing_pool(queries, api_base, api_key):
    """
    路由池管理 - 处理多查询并发路由
    支持模型并发调用，优化响应时间
    """

def check_llm_name(target_llm):
    """
    智能模型映射 - 将友好名称映射到API模型ID
    支持多厂商模型：OpenAI, Meta, Mistral, Google等
    """

def get_llm_response_via_api(prompt, LLM_MODEL, ...):
    """
    API调用封装 - 统一的模型调用接口
    包含重试机制、超时处理、错误恢复
    """
```

**支持的模型生态**:
- **Qwen系列**: qwen/qwen2.5-7b-instruct
- **LLaMA系列**: meta/llama-3.1-70b-instruct, meta/llama-3.1-8b-instruct
- **Mistral系列**: mistralai/mistral-7b-instruct-v0.3, mistralai/mixtral-8x22b-instruct
- **Gemma系列**: google/gemma-2-27b-it
- **专业模型**: nvidia/llama3-chatqa-1.5-8b, writer/palmyra-creative-122b

### 2. 多轮推理流程

**推理模式**:
```
<think>
内部推理 - 分析问题，确定需要的外部信息
</think>

<search>模型名:具体查询</search>
外部路由 - 调用专业模型获取信息

<information>
专家响应 - 集成外部模型的回答
</information>

<answer>
最终答案 - 综合所有信息给出结果
</answer>
```

**提示词模板** ([`prompt_pool.py`](./data_process/prompt_pool.py)):
- **PROMPT_TEMPLATE_QWEN**: 针对Qwen模型优化的4500+字符模板
- **PROMPT_TEMPLATE_LLAMA**: 针对LLaMA模型优化的推理模板
- 包含详细的模型描述和使用指南

### 3. 强化学习训练框架 (VERL)

**训练架构**:
```
VERL Framework
├── Actor (策略网络)
│   ├── 路由决策生成
│   ├── 模型选择策略
│   └── 多轮交互控制
├── Critic (价值网络)
│   ├── 状态价值估计
│   ├── 奖励信号评估
│   └── 策略优化指导
└── Rollout (推理引擎)
    ├── vLLM集成
    ├── 分布式推理
    └── 批处理优化
```

**核心训练组件** ([`verl/`](./verl/)):
- **[`trainer/main_ppo.py`](./verl/trainer/main_ppo.py)**: PPO算法实现
- **[`workers/`](./verl/workers/)**: Actor, Critic, Rollout工作器
- **[`models/`](./verl/models/)**: 支持LLaMA, Qwen等模型
- **[`utils/`](./verl/utils/)**: 分布式训练、日志、评估工具

### 4. 奖励机制设计

**多层次奖励函数**:
```python
total_reward = format_reward + outcome_reward + cost_reward

# 格式奖励: 确保输出符合<think><search><answer>格式
# 结果奖励: 基于Exact Match或F1-Score评估最终答案质量  
# 成本奖励: 平衡性能与API调用成本，支持cost_coe参数调节
```

**成本计算** ([`API_PRICE_1M_TOKENS`](./router_r1/llm_agent/route_service.py)):
```python
API_PRICE_1M_TOKENS = {
    "qwen/qwen2.5-7b-instruct": 0.3,
    "meta/llama-3.1-70b-instruct": 0.88,
    "meta/llama-3.1-8b-instruct": 0.18,
    # ... 更多模型定价
}
```

## 🚀 技术特性

### 分布式训练支持
- **多GPU训练**: 支持tensor并行和数据并行
- **FSDP优化**: 参数、梯度、优化器卸载
- **vLLM集成**: 高效的推理加速
- **Ray分布式**: 支持多节点扩展

### 模型适配性
- **Transformer架构**: 支持主流开源模型
- **API统一接口**: OpenAI API兼容
- **动态batch**: 优化GPU利用率
- **梯度检查点**: 降低显存占用

### 评估指标
- **Exact Match (EM)**: 精确匹配评估
- **F1-Score**: 语义相似度评估
- **Cost Efficiency**: 成本效益分析
- **Response Quality**: 响应质量评估

## 📊 数据流架构

### 训练数据流
```
原始QA数据集 → 数据预处理 → 多轮对话构造 → 训练样本生成
     ↓              ↓              ↓              ↓
NQ, HotpotQA → qa_train_merge → 路由标注 → PPO训练数据
```

### 推理数据流  
```
用户查询 → 初始提示词 → 多轮路由 → 结果聚合 → 最终答案
    ↓         ↓          ↓         ↓         ↓
Question → Template → Route → Integrate → Answer
```

## 🎯 核心优势

### 1. 智能路由策略
- **任务自适应**: 根据问题类型自动选择最优模型组合
- **成本优化**: 平衡性能和API调用成本
- **并发处理**: 支持多查询并行路由

### 2. 多轮协作机制
- **信息累积**: 每轮推理结果作为下轮输入
- **专家协作**: 不同模型贡献各自专长
- **动态终止**: 智能判断何时给出最终答案

### 3. 强化学习优化
- **策略学习**: 通过RL学习最优路由策略
- **奖励塑造**: 多维度奖励函数指导训练
- **在线优化**: 支持持续学习和策略改进

## 🔧 配置与扩展

### 自定义模型池
1. **模型描述配置**: 在[`prompt_pool.py`](./data_process/prompt_pool.py)中添加新模型描述
2. **路由映射更新**: 修改[`check_llm_name`](./router_r1/llm_agent/route_service.py)函数
3. **定价配置**: 更新[`API_PRICE_1M_TOKENS`](./router_r1/llm_agent/route_service.py)字典
4. **API集成**: 配置相应的API端点和密钥

### 训练参数调优
```bash
# 性能vs成本权衡
cost_coe=0.9  # 高成本敏感度

# 奖励指标选择  
reward_metric="em"    # 精确匹配
reward_metric="f1"    # F1分数

# 批处理优化
train_batch_size=64   # 建议使用大批次以提升稳定性
```

## 📚 相关研究

Router-R1基于以下研究基础：
- **强化学习**: PPO算法优化路由策略
- **多智能体系统**: 模型协作机制设计  
- **课程学习**: 渐进式训练策略
- **成本效益优化**: 性能-成本平衡算法

## 🏆 应用场景

- **复杂问答**: 多跳推理、知识整合
- **专业咨询**: 法律、医疗、技术领域
- **创意生成**: 多角度创意协作
- **代码开发**: 多语言编程支持
- **学术研究**: 文献综述、论文写作

---

**总结**: Router-R1 通过创新的多轮路由架构和强化学习优化，实现了大型语言模型的智能协作，在复杂任务上显著超越单模型性能，同时优化了成本效益。这一架构为未来的多模型系统设计提供了重要参考。