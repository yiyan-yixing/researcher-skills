"""
隐藏状态提取模块：加载模型，对每个样本推理并提取每层最后一个 token 的隐藏状态。

输出：
  - results/{run_id}/hidden_states.json  — 每层隐藏状态 + 样本元信息
"""

import argparse
import json
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(cfg_model):
    """加载 HuggingFace 模型和 tokenizer。"""
    print(f"[Model] Loading {cfg_model['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg_model["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg_model["name"],
        torch_dtype=getattr(torch, cfg_model.get("torch_dtype", "float32")),
    )
    model.to(cfg_model.get("device", "cpu"))
    model.eval()

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
    hidden_size = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size

    print(f"[Model] Loaded: {cfg_model['name']}, {num_layers} layers, hidden_size={hidden_size}")
    return model, tokenizer, num_layers, hidden_size


def extract_hidden_states(model, tokenizer, samples, cfg_data, cfg_model):
    """
    对每个样本提取每层最后一个 token 的隐藏状态。

    返回:
        results: list of dict, 每个 dict 包含样本 id、benchmark、label、每层隐藏状态(numpy)
    """
    device = cfg_model.get("device", "cpu")
    max_length = cfg_data.get("max_input_length", 512)
    num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers

    all_results = []

    for idx, sample in enumerate(samples):
        input_text = sample["input_text"]

        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass with output_hidden_states=True
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 提取每层最后一个 token 的隐藏状态
        # hidden_states 是一个 tuple, 长度 = num_layers + 1 (包含 embedding 层)
        hidden_states = outputs.hidden_states
        last_token_idx = inputs["input_ids"].shape[1] - 1

        layer_hidden = {}
        for layer_idx in range(num_layers):
            # layer_idx 0 对应 embedding 层, 从 1 开始是 transformer 层
            # 但 hidden_states[0] 是 embedding 层输出
            # hidden_states[1] 到 hidden_states[num_layers] 是各 transformer 层
            # 我们用 layer_idx 从 0 开始计数 transformer 层
            hs = hidden_states[layer_idx + 1]  # 跳过 embedding 层
            last_token_hidden = hs[0, last_token_idx, :].detach().cpu().tolist()
            layer_hidden[str(layer_idx)] = last_token_hidden

        result = {
            "id": sample["id"],
            "benchmark": sample["benchmark"],
            "category": sample["category"],
            "label": sample["label"],
            "hidden_states": layer_hidden,
        }
        all_results.append(result)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(samples)}] Extracted hidden states")

    return all_results, num_layers


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier for output directory")
    parser.add_argument("--samples-file", type=str, default=None, help="Path to all_samples.json")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_model = config["model"]
    cfg_data = config["data"]

    # 确定 run_id 和输出目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_id = args.run_id or f"run_{int(time.time())}"
    results_dir = os.path.join(base_dir, config["output"]["results_dir"], run_id)
    os.makedirs(results_dir, exist_ok=True)

    # 加载样本数据
    samples_file = args.samples_file or os.path.join(base_dir, "data", "all_samples.json")
    print(f"[Data] Loading samples from {samples_file}")
    with open(samples_file, "r") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"[Data] Loaded {len(samples)} samples")

    # 加载模型
    model, tokenizer, num_layers, hidden_size = load_model_and_tokenizer(cfg_model)

    # 提取隐藏状态
    print(f"[Extract] Extracting hidden states for {len(samples)} samples...")
    start_time = time.time()
    results, num_layers = extract_hidden_states(model, tokenizer, samples, cfg_data, cfg_model)
    elapsed = time.time() - start_time
    print(f"[Extract] Done in {elapsed:.1f}s")

    # 保存结果
    output_path = os.path.join(results_dir, "hidden_states.json")
    # 添加模型元信息
    output_data = {
        "model": cfg_model["name"],
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_samples": len(results),
        "extraction_time_s": round(elapsed, 1),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f)
    print(f"[Save] Hidden states saved to {output_path}")

    # 如果隐藏状态太大，也保存一个精简版（不含隐藏状态向量，仅元信息）
    if config["output"].get("save_hidden_states", True):
        meta_path = os.path.join(results_dir, "hidden_states_meta.json")
        meta_data = {
            "model": cfg_model["name"],
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_samples": len(results),
            "sample_ids": [r["id"] for r in results],
            "sample_benchmarks": [r["benchmark"] for r in results],
            "sample_labels": [r["label"] for r in results],
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

    return output_path


if __name__ == "__main__":
    main()
