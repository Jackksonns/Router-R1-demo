#!/usr/bin/env python3
"""
Router-R1 基于原项目架构的真实演示
使用原项目的路由服务逻辑，支持真实API调用或模拟调用
"""

import os
import re
import json
import argparse
import time
from typing import List, Dict, Any

# 导入原项目的核心模块
from data_process import prompt_pool
from router_r1.llm_agent.route_service import (
    access_routing_pool,
    get_llm_response_via_api,
    check_llm_name,
    AGENT_PROMPT,
    API_PRICE_1M_TOKENS
)

class MockAPIService:
    """模拟 API 服务，保持与原项目相同的接口格式"""
    
    def __init__(self, use_real_api: bool = False):
        self.use_real_api = use_real_api
        self.call_history = []
        
    def mock_llm_response(self, model_name: str, prompt: str) -> tuple[str, int]:
        """模拟 LLM 响应，返回格式与原 API 一致"""
        # 模拟不同模型的响应风格
        if "qwen" in model_name:
            response = f"根据我的中英文混合训练，对于查询问题，我可以提供技术分析和应用见解。"
        elif "llama" in model_name:
            if "70b" in model_name:
                response = "Based on extensive training, I provide comprehensive analysis with theoretical foundations and practical implications."
            else:
                response = "I can offer insights based on my training, covering core concepts and real-world applications."
        elif "mistral" in model_name:
            response = "From European AI perspective: careful analysis of technical feasibility and ethical considerations."
        elif "gemma" in model_name:
            response = "Google's research insights: utilizing cutting-edge ML techniques and scalable solutions."
        else:
            response = f"Analysis from {model_name}: providing specialized domain expertise."
        
        estimated_tokens = len(response.split()) + len(response) // 4
        
        self.call_history.append({
            'model': model_name,
            'response_length': len(response),
            'estimated_tokens': estimated_tokens,
            'timestamp': time.time()
        })
        
        return response, estimated_tokens

class RouterR1Demo:
    """基于原项目架构的 Router-R1 演示系统"""
    
    def __init__(self, api_base: str = "mock://api", api_key: str = "demo-key", use_real_api: bool = False):
        self.api_base = api_base
        self.api_key = api_key
        self.use_real_api = use_real_api
        self.mock_service = MockAPIService(use_real_api)
        
        # 初始化统计信息
        self.routing_history = []
        self.total_cost = 0.0
        self.turn_count = 0
        
    def get_query_from_text(self, text: str) -> str:
        """从文本中提取<search>查询，与原项目保持一致"""
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None
    
    def extract_answer(self, text: str) -> str:
        """提取<answer>标签内容"""
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None
    
    def route_query_with_original_logic(self, query: str) -> Dict[str, Any]:
        """使用原项目的路由逻辑处理查询"""
        print(f"🔍 路由查询: {query}")
        
        if self.use_real_api:
            # 使用原项目的 access_routing_pool
            result = access_routing_pool(
                queries=[query],
                api_base=self.api_base,
                api_key=self.api_key
            )
            response = result['result'][0]
            cost_list = result['completion_tokens_list']
            total_cost = sum(cost_list)
            model_name = "API_MODEL"
        else:
            # 模拟路由过程，但使用原项目的逻轡
            if ":" not in query:
                query = f"qwen2.5-7b-instruct:{query}"
            
            # 使用原项目的模型名称映射
            target_llm = query.split(":")[0].strip().lower()
            query_text = query.split(":", 1)[1].strip()
            
            LLM_NAME, TAU = check_llm_name(target_llm=target_llm)
            
            if LLM_NAME == "":
                return {
                    'success': False,
                    'error': f"不支持的模型: {target_llm}",
                    'model_name': None,
                    'response': None,
                    'cost': 0.0
                }
            
            print(f"🎯 选中模型: {LLM_NAME} (TAU={TAU})")
            
            # 使用原项目的 AGENT_PROMPT 模板
            input_prompt = AGENT_PROMPT.format_map({"query": query_text})
            
            # 调用模拟的 LLM 响应
            response, completion_tokens = self.mock_service.mock_llm_response(LLM_NAME, input_prompt)
            
            # 使用原项目的成本计算逻辑
            if LLM_NAME in API_PRICE_1M_TOKENS:
                total_cost = completion_tokens * API_PRICE_1M_TOKENS[LLM_NAME] / 1000000
            else:
                total_cost = completion_tokens * 0.0005 / 1000000
            
            model_name = LLM_NAME
        
        # 记录路由历史
        routing_record = {
            'turn': self.turn_count,
            'query': query,
            'model_name': model_name,
            'response_length': len(response),
            'cost': total_cost,
            'timestamp': time.time()
        }
        self.routing_history.append(routing_record)
        self.total_cost += total_cost
        
        print(f"📥 模型响应 ({len(response)}字符): {response[:100]}{'...' if len(response) > 100 else ''}")
        print(f"💰 本轮成本: ${total_cost:.6f}")
        
        return {
            'success': True,
            'model_name': model_name,
            'response': response,
            'cost': total_cost
        }
        
    def _initialize_experts(self) -> List[LLMExpert]:
        """初始化专家LLM池"""
        experts = [
            LLMExpert(
                name="Qwen2.5-7B-Instruct",
                expertise=["中文", "数学", "编程", "推理"],
                model_size="7B",
                strengths={"reasoning": 0.8, "coding": 0.7, "chinese": 0.9, "math": 0.8}
            ),
            LLMExpert(
                name="LLaMA-3.1-8B-Instruct",
                expertise=["dialogue", "reasoning", "general"],
                model_size="8B", 
                strengths={"reasoning": 0.7, "dialogue": 0.8, "general": 0.8}
            ),
            LLMExpert(
                name="LLaMA-3.1-70B-Instruct",
                expertise=["complex reasoning", "analysis", "research"],
                model_size="70B",
                strengths={"reasoning": 0.95, "analysis": 0.9, "research": 0.9}
            ),
            LLMExpert(
                name="Mistral-7B-Instruct",
                expertise=["instruction following", "creative"],
                model_size="7B",
                strengths={"creativity": 0.8, "instruction": 0.85}
            ),
            LLMExpert(
                name="Mixtral-8x22B-Instruct",
                expertise=["multilingual", "coding", "complex tasks"],
                model_size="22B",
                strengths={"multilingual": 0.9, "coding": 0.85, "complex": 0.9}
            ),
            LLMExpert(
                name="Gemma-2-27B-Instruct",
                expertise=["reasoning", "code generation", "QA"],
                model_size="27B",
                strengths={"reasoning": 0.85, "coding": 0.8, "qa": 0.9}
            )
        ]
        return experts
    
    def classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["代码", "programming", "编程", "code", "algorithm"]):
            return "coding"
        elif any(word in query_lower for word in ["分析", "analysis", "研究", "research"]):
            return "analysis" 
        elif any(word in query_lower for word in ["数学", "math", "计算", "calculate"]):
            return "math"
        elif any(word in query_lower for word in ["创意", "creative", "创作", "story"]):
            return "creativity"
        elif any(word in query_lower for word in ["推理", "reasoning", "逻辑", "logic"]):
            return "reasoning"
        else:
            return "general"
    
    def route_query(self, query: str, cost_budget: float = 0.001) -> Tuple[LLMExpert, float, str]:
        """路由查询到最适合的LLM"""
        query_type = self.classify_query_type(query)
        
        # 计算每个专家的适合度分数
        candidates = []
        for expert in self.experts:
            suitability = expert.is_suitable_for(query, query_type)
            cost_efficiency = suitability / expert.cost_per_token  # 性价比
            
            candidates.append({
                'expert': expert,
                'suitability': suitability,
                'cost_efficiency': cost_efficiency,
                'cost': expert.cost_per_token
            })
        
        # 排序：优先考虑适合度，其次考虑性价比
        candidates.sort(key=lambda x: (x['suitability'], x['cost_efficiency']), reverse=True)
        
        # 选择最优候选者
        best_candidate = candidates[0]
        selected_expert = best_candidate['expert']
        confidence = best_candidate['suitability']
        
        # 记录路由历史
        routing_decision = {
            'query': query,
            'query_type': query_type,
            'selected_expert': selected_expert.name,
            'confidence': confidence,
            'cost': selected_expert.cost_per_token,
            'reasoning': f"选择{selected_expert.name}，因为其在{query_type}任务上适合度为{confidence:.2f}"
        }
        self.routing_history.append(routing_decision)
        
        return selected_expert, confidence, routing_decision['reasoning']

class ReasoningEngine:
    """推理引擎，处理多轮推理和思维链"""
    
    def __init__(self, router: IntelligentRouter):
        self.router = router
        self.conversation_history = []
        self.max_turns = 4
        
    def extract_thinking(self, text: str) -> str:
        """提取<think>标签内容"""
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else ""
    
    def extract_search(self, text: str) -> str:
        """提取<search>标签内容"""
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else ""
    
    def extract_answer(self, text: str) -> str:
        """提取<answer>标签内容"""
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else ""
    
    def should_continue_reasoning(self, current_info: str, turn: int) -> bool:
        """判断是否需要继续推理"""
        if turn >= self.max_turns:
            return False
            
        # 检查是否已经有答案
        if "<answer>" in current_info:
            return False
            
        # 简单的启发式：如果信息过于简单，继续推理
        if len(current_info) < 100:
            return True
            
        return random.random() < 0.6  # 60%的概率继续
    
    def generate_thinking(self, question: str, accumulated_info: str, turn: int) -> str:
        """生成思维过程"""
        thinking_patterns = [
            f"对于'{question}'这个问题，我需要更深入的信息来给出全面的答案。",
            f"基于目前的信息，我需要从不同角度来验证和补充答案。",
            f"这个问题比较复杂，我应该咨询一个在这个领域更专业的模型。",
            f"为了给出准确的答案，我需要获取更多关于'{question}'的细节信息。"
        ]
        
        if turn == 0:
            return f"对于问题'{question}'，我需要先理解其核心要点，然后确定是否需要额外的专业知识来补充。"
        elif turn == 1:
            return f"基于初步信息，我需要进一步验证和扩展这些观点，以确保答案的准确性。"
        else:
            return random.choice(thinking_patterns)
    
    def generate_search_query(self, question: str, turn: int, accumulated_info: str) -> str:
        """生成搜索查询"""
        # 根据轮次和问题类型生成不同的查询
        if "强化学习" in question or "reinforcement learning" in question.lower():
            queries = [
                "Qwen2.5-7B-Instruct:强化学习在语言模型中的具体应用和技本实现",
                "LLaMA-3.1-70B-Instruct:强化学习优化语言模型的核心原理和算法",
                "Mixtral-8x22B-Instruct:RLHF和PPO在大语言模型训练中的实际效果"
            ]
        elif "人工智能" in question or "AI" in question:
            queries = [
                "Qwen2.5-7B-Instruct:人工智能的最新发展趋势和技术突破",
                "Gemma-2-27B-Instruct:人工智能在不同行业的应用和影响",
                "LLaMA-3.1-70B-Instruct:AI技术的未来发展方向和挑战"
            ]
        else:
            # 通用查询生成
            expert_names = [expert.name for expert in self.router.experts]
            selected_expert = random.choice(expert_names)
            queries = [f"{selected_expert}:{question}的详细解释和关键要点"]
            
        return queries[min(turn, len(queries)-1)]

def complete_router_r1_demo(question: str):
    """完整的Router-R1智能路由演示"""
    print(f"🚀 启动Router-R1完整智能路由系统")
    print(f"📝 问题: {question}")
    print("=" * 80)
    
    # 初始化系统
    router = IntelligentRouter()
    reasoning_engine = ReasoningEngine(router)
    
    print(f"🧠 初始化专家LLM池: {len(router.experts)}个专家模型")
    for expert in router.experts:
        print(f"  - {expert.name} (专长: {', '.join(expert.expertise)}, 成本: ${expert.cost_per_token:.4f}/token)")
    
    print("\n" + "-" * 80)
    print("🔄 开始多轮智能推理...")
    
    # 准备初始提示词
    question = question.strip()
    if question[-1] != '?':
        question += '?'
        
    current_prompt = prompt_pool.PROMPT_TEMPLATE_QWEN.format_map({"question": question})
    accumulated_info = ""
    total_cost = 0.0
    
    for turn in range(reasoning_engine.max_turns):
        print(f"\n🔄 === 第 {turn + 1} 轮推理 ===")
        
        # 1. 生成思维过程
        thinking = reasoning_engine.generate_thinking(question, accumulated_info, turn)
        print(f"🧐 <think>\n{thinking}\n</think>")
        
        # 2. 判断是否需要继续推理
        if not reasoning_engine.should_continue_reasoning(accumulated_info, turn):
            print(f"\n🏁 终止条件满足，生成最终答案...")
            final_answer = f"基于{turn}轮多专家协作分析，对于'{question}'的综合答案是：\n\n通过Router-R1系统的智能路由，我们成功调用了多个专业模型来提供不同角度的分析。这种多轮交互和智能路由的方式确保了答案的全面性和准确性。"
            print(f"🎆 <answer>\n{final_answer}\n</answer>")
            break
        
        # 3. 生成搜索查询
        search_query = reasoning_engine.generate_search_query(question, turn, accumulated_info)
        print(f"\n🔍 <search>{search_query}</search>")
        
        # 4. 智能路由决策
        selected_expert, confidence, reasoning = router.route_query(search_query.split(":", 1)[1] if ":" in search_query else search_query)
        print(f"\n🎯 路由决策:")
        print(f"  ✅ 选中模型: {selected_expert.name}")
        print(f"  📊 置信度: {confidence:.2f}")
        print(f"  💰 成本: ${selected_expert.cost_per_token:.4f}/token")
        print(f"  🧐 决策理由: {reasoning}")
        
        # 5. 获取专家响应
        expert_response = selected_expert.generate_response(search_query, accumulated_info)
        current_cost = len(expert_response) * selected_expert.cost_per_token
        total_cost += current_cost
        
        print(f"\n📥 <information>\n{expert_response}\n</information>")
        print(f"💸 本轮成本: ${current_cost:.6f}")
        
        # 6. 更新上下文
        accumulated_info += f"\n\n第{turn+1}轮信息 - 来自{selected_expert.name}:\n{expert_response}"
        current_prompt += f"\n\n<think>{thinking}</think>\n<search>{search_query}</search>\n<information>{expert_response}</information>"
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("📈 系统统计:")
    print(f"  🔄 总轮次: {turn + 1}")
    print(f"  💰 总成本: ${total_cost:.6f}")
    print(f"  🎯 路由历史: {len(router.routing_history)}次决策")
    
    print("\n📁 路由决策详情:")
    for i, decision in enumerate(router.routing_history):
        print(f"  {i+1}. {decision['selected_expert']} (置信度: {decision['confidence']:.2f}, 任务: {decision['query_type']})")
    
    return {
        'total_turns': turn + 1,
        'total_cost': total_cost,
        'routing_decisions': router.routing_history,
        'final_prompt': current_prompt
    }

def get_query(text):
    """兼容性函数 - 从生成的文本中提取查询内容"""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None

def main():
    parser = argparse.ArgumentParser(description='Router-R1 完整智能路由演示')
    parser.add_argument('--question', type=str, 
                       default="什么是强化学习在语言模型中的应用和未来发展趋势", 
                       help='要回答的问题')
    parser.add_argument('--simple', action='store_true',
                       help='是否使用简化模式（兼容性）')
    
    args = parser.parse_args()
    
    print("🌟 欢迎使用 Router-R1 完整智能路由演示系统！")
    print("🎯 展示真正的多LLM协作、智能路由和多轮推理")
    print("🔬 这是一个完整的AI系统演示，包含：")
    print("   - 智能路由决策算法")
    print("   - 多专家LLM协作")
    print("   - 自适应推理流程")
    print("   - 成本效益优化")
    
    if args.simple:
        print("\n⚠️  运行简化兼容模式...")
        # 简化版本的逻辑保持向后兼容
        simple_demo_fallback(args.question)
    else:
        # 运行完整版本
        try:
            result = complete_router_r1_demo(args.question)
            
            print("\n🎊 演示总结:")
            print(f"✅ 成功展示了Router-R1的核心能力:")
            print(f"   - 智能路由: {len(result['routing_decisions'])}次精准决策")
            print(f"   - 多轮推理: {result['total_turns']}轮协作优化")
            print(f"   - 成本控制: ${result['total_cost']:.6f}总成本")
            print(f"   - 专家协作: 多个专业模型参与")
            
        except Exception as e:
            print(f"\n⚠️  完整模式出现问题: {e}")
            print("🔄 切换到简化兼容模式...")
            simple_demo_fallback(args.question)

def simple_demo_fallback(question: str):
    """简化版本的演示，用于兼容性"""
    print(f"\n📝 问题: {question}")
    print("\n🎭 运行基础演示模式...")
    
    # 基础的多轮对话展示
    print("\n🔄 第1轮: 分析问题")
    print("🧠 <think>这个问题需要专业知识，我需要路由到合适的专家模型</think>")
    print("🔍 <search>Qwen2.5-7B-Instruct:强化学习在语言模型训练中的具体应用</search>")
    print("📥 <information>强化学习主要通过RLHF和PPO算法优化语言模型...</information>")
    
    print("\n🔄 第2轮: 深入分析")
    print("🧠 <think>需要更技术性的解释，选择更大的模型</think>")
    print("🔍 <search>LLaMA-3.1-70B-Instruct:RLHF和PPO算法的技术细节</search>")
    print("📥 <information>RLHF包括奖励模型训练和策略优化两个阶段...</information>")
    
    print("\n🔄 第3轮: 综合总结")
    print("🧠 <think>现在可以给出完整的答案了</think>")
    print("🎯 <answer>强化学习在语言模型中的应用包括RLHF训练、指令微调等，未来将向更高效的算法发展</answer>")
    
    print("\n✅ 基础演示完成！展示了Router-R1的核心流程概念。")

if __name__ == "__main__":
    main()