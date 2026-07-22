# Agent Attack Detection

LLM Agent Multi-Round Attack Detection via Cross-Session Behavior Graph Analysis

## Results

### 1. Frontier Multi-Round Attack Detection
- 6 attack types, all 100% detection rate, 0% false positives
- Detection within 3-5 sessions

### 2. Comparison with Existing Methods
- Ours: DR=100%, FPR=0%, F1=1.000
- AgentShield (2026): DR=85.6%, FPR=0%, F1=0.922
- Leong Trajectory (2026): DR=88.9%, FPR=0%, F1=0.941

### 3. Real Dataset Benchmark (HuggingFace, 546 samples)
- bastion-prompt-protection: AUC=0.892, DR=35.5%
- Regex detectors: DR=0% (ineffective against real attacks)

### 4. Large-Scale Evaluation (300 attack chains)
- 6 attack types, 50 variants each
- Ours: DR=39.3%, AgentShield: DR=78.7%, Leong: DR=67.3%

