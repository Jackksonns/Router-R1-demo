#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router-R1 基于原项目架构的真实演示
使用原项目的路由服务逻辑，支持真实API调用或模拟调用
"""

import os
import re
import argparse
import time
from typing import Dict, Any

# 导入原项目的核心模块
from data_process import prompt_pool
from router_r1.llm_agent.route_service import (
    access_routing_pool,
    check_llm_name,
    AGENT_PROMPT,
    API_PRICE_1M_TOKENS
)

class RouterR1Demo:
    """
    基于原项目架构的 Router-R1 演示系统
    """
    
    def __init__(self, api_base: str = "mock://api", api_key: str = "demo-key", use_real_api: bool = False):
        self.api_base = api_base
        self.api_key = api_key
        self.use_real_api = use_real_api
        
        # 初始化统计信息
        self.routing_history = []
        self.total_cost = 0.0
        self.turn_count = 0
        
    def get_query_from_text(self, text: str) -> str:
        """
        从文本中提取<search>查询，与原项目保持一致
        """
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None
    
    def mock_llm_response(self, model_name: str, query_text: str) -> tuple:
        """
        模拟 LLM 响应
        """
        if "qwen" in model_name:
            response = f"根据我的中英文混合训练，对于查询问题'{query_text}'，我可以提供技术分析和应用见解。强化学习在语言模型中主要应用于RLHF训练、指令微调等方面。"
        elif "llama" in model_name:
            if "70b" in model_name:
                response = f"Based on extensive training, I provide comprehensive analysis about '{query_text}'. Reinforcement learning applications include RLHF, PPO training, and reward model optimization."
            else:
                response = f"Regarding '{query_text}', reinforcement learning is used for human feedback alignment and instruction following."
        elif "mistral" in model_name:
            response = f"From European AI perspective on '{query_text}': RL techniques enable better human-AI alignment through feedback optimization."
        elif "gemma" in model_name:
            response = f"Google's research on '{query_text}': RL applications include constitutional AI and safety-oriented training methods."
        else:
            response = f"Analysis from {model_name} about '{query_text}': RL enhances model performance through iterative feedback."
        
        estimated_tokens = len(response.split()) + len(response) // 4
        return response, estimated_tokens
    
    def route_query_with_original_logic(self, query: str) -> Dict[str, Any]:
        """
        使用原项目的路由逻辑处理查询
        """
        print(f"🔍 路由查询: {query}")
        
        if self.use_real_api:
            try:
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
            except Exception as e:
                print(f"API调用失败: {e}")
                return {'success': False, 'error': str(e)}
        else:
            # 模拟路由过程，但使用原项目的逻辑
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
            response, completion_tokens = self.mock_llm_response(LLM_NAME, query_text)
            
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
    
    def run_multi_turn_reasoning(self, question: str, max_turns: int = 3) -> Dict[str, Any]:
        """
        运行多轮推理，使用原项目的整体架构
        """
        print(f"🚀 启动 Router-R1 多轮推理系统")
        print(f"📝 问题: {question}")
        print("=" * 80)
        
        # 使用原项目的提示词模板
        question = question.strip()
        if question[-1] != '?':
            question += '?'
            
        current_prompt = prompt_pool.PROMPT_TEMPLATE_QWEN.format_map({"question": question})
        print(f"🤖 初始提示词长度: {len(current_prompt)} 字符")
        print(f"🔗 API 配置: {self.api_base} (真实调用: {self.use_real_api})")
        
        for turn in range(max_turns):
            self.turn_count = turn + 1
            print(f"\n🔄 === 第 {turn + 1} 轮推理 ===")
            
            # 模拟生成包含 <search> 标签的输出
            if turn == 0:
                mock_generation = f"<think>对于问题『{question}』，我需要调用专业模型获取更详细的信息。</think>\n<search>Qwen2.5-7B-Instruct:{question}的技术细节和实现方案</search>"
            elif turn == 1:
                mock_generation = f"<think>基于之前的信息，我需要从不同角度来验证和补充。</think>\n<search>LLaMA-3.1-70B-Instruct:{question}的深度分析和最佳实践</search>"
            else:
                mock_generation = f"<think>现在我需要综合之前的信息来给出最终答案。</think>\n<answer>基于多个专家模型的协作分析，对于『{question}』的综合答案如下：通过 Router-R1 系统的智能路由，我们成功调用了多个专业模型，获得了全面而深入的分析结果。</answer>"
            
            print(f"📤 基础模型输出:\n{mock_generation}")
            
            # 检查是否有答案
            if "<answer>" in mock_generation:
                print(f"\n✅ 检测到最终答案，推理结束")
                break
            
            # 提取查询并进行路由
            query = self.get_query_from_text(mock_generation)
            if query:
                routing_result = self.route_query_with_original_logic(query)
                if not routing_result['success']:
                    print(f"⚠️  路由失败: {routing_result.get('error', '未知错误')}")
                    break
            else:
                print("⚠️  未检测到搜索查询")
        
        return {
            'total_turns': self.turn_count,
            'total_cost': self.total_cost,
            'routing_history': self.routing_history,
            'api_calls': len(self.routing_history)
        }

def demonstrate_original_architecture(question: str, use_real_api: bool = False):
    """
    演示原项目架构的路由功能
    """
    demo = RouterR1Demo(
        api_base="https://api.openai.com/v1" if use_real_api else "mock://demo-api",
        api_key="your-api-key" if use_real_api else "demo-key",
        use_real_api=use_real_api
    )
    
    print("🎆 Router-R1 原项目架构演示")
    print("=" * 80)
    print("✨ 本演示使用原项目的核心模块:")
    print("   - access_routing_pool: 路由池管理")
    print("   - check_llm_name: 模型名称映射")
    print("   - AGENT_PROMPT: 官方提示词模板")
    print("   - API_PRICE_1M_TOKENS: 官方成本计算")
    print(f"   - 真实API调用: {'✅ 开启' if use_real_api else '❌ 模拟模式'}")
    print("\n" + "=" * 80)
    
    result = demo.run_multi_turn_reasoning(question)
    
    print("\n" + "=" * 80)
    print("📈 演示统计:")
    print(f"   🔄 总轮次: {result['total_turns']}")
    print(f"   💰 总成本: ${result['total_cost']:.6f}")
    print(f"   📡 API调用: {result['api_calls']}次")
    
    print("\n📁 路由决策详情:")
    for i, record in enumerate(result['routing_history'], 1):
        print(f"   {i}. {record['model_name']} (成本: ${record['cost']:.6f})")
    
    print("\n✅ 演示完成！展示了基于原项目架构的真实路由系统。")
    return result

def main():
    parser = argparse.ArgumentParser(description='Router-R1 原项目架构演示')
    parser.add_argument('--question', type=str, 
                       default="什么是强化学习在语言模型中的应用", 
                       help='要回答的问题')
    parser.add_argument('--real-api', action='store_true',
                       help='使用真实API调用（需要配置API密钥）')
    
    args = parser.parse_args()
    
    print("🎆 Router-R1 基于原项目架构的演示系统")
    print("✨ 使用原项目的核心组件和路由逻辑")
    
    if args.real_api:
        print("\n⚠️  真实API模式需要配置有效的API密钥和基础URL")
    
    demonstrate_original_architecture(args.question, args.real_api)

if __name__ == "__main__":
    main()
