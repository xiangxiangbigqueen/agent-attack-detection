# Agent Attack Detection

多轮攻击检测系统，用于检测 LLM 智能体上的多轮和延迟触发攻击。

## 实验结果

| 方法 | 检测率 | 误报率 |
|------|:-----:|:-----:|
| 单规则 Baseline | 33.3% | 0.0% |
| 本文方法（最优阈值） | **100%** | **0%** |

## 项目结构

```
agent_attack_detection/
├── agent/          # 智能体系统
├── attack/         # 攻击场景
├── detection/      # 检测方法（核心）
├── experiments/    # 实验代码
└── main.py         # 入口
```

## 快速开始

```bash
python experiments/evaluation.py    # 跑实验
python main.py --quick              # 快速测试
```

## 方法简介

跨轮次行为图谱检测。不依赖特定工具名，自动学习正常行为模式，累计异常分跨会话追踪。

## 引用

```
DSC 2026, Multi-Round Attack Detection for LLM Agents via Cross-Session Behavior Graph Analysis
```
