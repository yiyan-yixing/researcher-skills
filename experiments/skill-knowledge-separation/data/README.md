# 数据目录说明

## Mini-Benchmark 数据

本实验使用 4 个 mini-benchmark，每个采样 100 题：

| Benchmark | 类型 | 来源 | 说明 |
|-----------|------|------|------|
| GSM8K-mini | 技能 | `gsm8k` (main/test) | 数学推理，需要多步计算 |
| IFEval-mini | 技能 | `google/IFEval` (train) | 指令遵循，需要格式控制能力 |
| MMLU-mini | 知识 | `cais/mmlu` (all/test) | 世界知识，依赖事实记忆 |
| TriviaQA-mini | 知识 | `trivia_qa` (rc.nocontext/validation) | 问答，依赖事实检索 |

## 数据获取

数据通过 `prepare_data.py` 自动下载和采样，无需手动准备：

```bash
python data/prepare_data.py --config config.yaml
```

## 分类标签

- **技能 (skill)**: GSM8K, IFEval → label = 1
- **知识 (knowledge)**: MMLU, TriviaQA → label = 0

## 数据格式

`prepare_data.py` 输出 JSON 文件到 `data/` 目录，格式：

```json
{
  "samples": [
    {
      "id": "gsm8k_0",
      "benchmark": "gsm8k",
      "category": "skill",
      "label": 1,
      "input_text": "Janet's ducks lay 16 eggs...",
      "answer": "18",
      "choices": null
    }
  ],
  "metadata": {
    "num_skill": 200,
    "num_knowledge": 200,
    "seed": 42
  }
}
```
