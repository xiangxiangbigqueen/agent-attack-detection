# Agent Attack Detection

检测 LLM 智能体上的多轮攻击（multi-round）和延迟触发攻击（delayed-trigger）。  
针对现有方法只能检测单轮攻击的不足，提出基于**跨轮次行为图谱**和**累计异常分**的检测方法。

## 用途

LLM 智能体在执行任务时会调用一系列工具（读邮件、转账、发消息等）。  
攻击者可以通过提示注入、记忆投毒等方式让智能体执行恶意操作。

**现存问题**：现有检测方法每次只看单次工具调用是否异常，但多轮攻击的每一步单独看都是合法操作，组合起来才是攻击。

**本方法**：记录完整的工具调用序列，构建跨轮次的行为图谱，通过累计异常分跨会话追踪，检测单步无法发现的组合攻击。

## 效果

| 方法 | 检测率 | 误报率 |
|------|:-----:|:-----:|
| 单规则 Baseline | 33.3% | 0.0% |
| 本文方法（最优阈值 0.8） | **100.0%** | **0.0%** |

覆盖 5 种攻击类型：延迟触发、多轮链式、跨会话记忆投毒、工具滥用、提示注入。

## 实验

运行以下命令即可复现全部实验结果：

```bash
python experiments/evaluation.py
```

输出包含 4 组实验：主实验对比、消融实验、攻击类型分析、阈值扫描。

## 项目结构

```
agent_attack_detection/
├── agent/core.py              # 智能体 + 记忆 + 工具调用
├── attack/scenarios.py        # 攻击场景定义
├── detection/graph_detector.py # 检测方法（行为图谱 + 累计分）
├── experiments/evaluation.py  # 实验评估
├── main.py                    # 入口
```

## 引用

```
DSC 2026, Multi-Round Attack Detection for LLM Agents via Cross-Session Behavior Graph Analysis
```
