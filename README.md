# Agent Attack Detection

多轮攻击检测系统，用于检测 LLM 智能体上的多轮和延迟触发攻击。

## 项目说明

LLM 智能体在执行任务时会调用一系列工具（读邮件、转账、发消息等）。攻击者可以通过提示注入、记忆投毒等方式让智能体执行恶意操作。

现有检测方法每次只看单次工具调用是否异常，但多轮攻击的每一步单独看都是合法操作，组合起来才是攻击。

本方法记录完整的工具调用序列，构建跨轮次的行为图谱，通过累计异常分跨会话追踪，检测单步无法发现的组合攻击。

## 已完成实验

### 1. 多轮攻击检测对比实验
`experiments/full_comparison.py`

9 个攻击场景（延迟触发、多轮链式、跨会话记忆投毒、工具滥用、提示注入），4 种方法对比：

| 方法 | 检测率 | 误报率 | F1 |
|:----|:-----:|:-----:|:--:|
| 本文方法（行为图谱） | **100%** | **0%** | **1.000** |
| AgentShield (2026) | 85.6% | 0% | 0.922 |
| Leong Trajectory (2026) | 88.9% | 0% | 0.941 |
| Random Baseline | 100% | 71.4% | 0.783 |

### 2. 单轮检测实验（自定义数据集）
`experiments/single_round_detection.py`

10 个良性 prompt + 10 个攻击 prompt：

| 检测器 | 检测率 | 误报率 | F1 | 延迟 |
|:------|:-----:|:-----:|:--:|:---:|
| bastion-prompt-protection (2026) | **100%** | 20% | 0.909 | 4s |
| nukon-pi-detect | 0% | 0% | 0 | 2ms |
| prompt-injection-sanitizer | 0% | 0% | 0 | 3ms |

### 3. 完整检测基准测试（真实数据集）
`experiments/complete_benchmark.py`

使用 HuggingFace deepset/prompt-injections 数据集（546 条样本），评估真实场景检测性能：

| 检测器 | 检测率 | 误报率 | AUC | 延迟 |
|:------|:-----:|:-----:|:--:|:---:|
| bastion-prompt-protection v1.3.5 | **35.5%** | **0.9%** | **0.892** | 60ms |
| prompt-injection-sanitizer | 0% | 0% | 0.500 | 3ms |
| nukon-pi-detect | 0% | 0% | 0.500 | 3ms |

### 4. 消融实验
`experiments/evaluation.py`

| 变体 | 检测率 | 误报率 | 结论 |
|:----|:-----:|:-----:|:----|
| 完整系统 | 55.6% | 0% | 基线 |
| 去掉跨轮次窗口 | **0%** | 0% | 图结构是核心 |
| 高衰减（不跨会话） | **0%** | 0% | 跨会话追踪必要 |

## 项目结构

```
agent_attack_detection/
├── agent/            # 智能体（API + 本地模型）
├── attack/           # 攻击场景（5种类型9个场景）
├── detection/        # 检测方法（核心）
│   └── baselines.py  # AgentShield + Leong 对比实现
├── experiments/      # 实验代码
│   ├── full_comparison.py      # 多轮对比实验
│   ├── single_round_detection.py # 单轮检测实验
│   ├── complete_benchmark.py   # 完整基准测试
│   └── evaluation.py           # 消融实验
├── data/             # 实验结果 JSON
└── main.py           # 入口
```

## 快速开始

```bash
# 使用 Python 3.11 环境
C:/Users/28995/agent_detect_env/Scripts/python experiments/full_comparison.py
C:/Users/28995/agent_detect_env/Scripts/python experiments/complete_benchmark.py
```

## 运行环境

- Python 3.11 虚拟环境：`C:/Users/28995/agent_detect_env`
- 已安装：agentdojo, openai, bastion-prompt-protection, numpy, pandas, networkx, pytorch 等
- 网络代理：Clash for Windows + ccswitch
