"""
共享工具函数：模型加载、模型架构适配、种子设定等。
"""

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """设置所有随机种子以保证复现性。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(cfg_model):
    """加载 HuggingFace 模型和 tokenizer，返回 (model, tokenizer, num_layers, hidden_size)。"""
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

    # 获取层数和隐藏维度（兼容 GPT-2 和 Llama 等不同架构）
    num_layers = get_num_layers(model)
    hidden_size = get_hidden_size(model)

    print(f"[Model] Loaded: {cfg_model['name']}, {num_layers} layers, hidden_size={hidden_size}")
    return model, tokenizer, num_layers, hidden_size


def get_num_layers(model):
    """获取模型的 transformer 层数。"""
    config = model.config
    if hasattr(config, "n_layer"):
        return config.n_layer
    elif hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    else:
        raise ValueError(f"Cannot determine number of layers from config: {config}")


def get_hidden_size(model):
    """获取模型的隐藏维度。"""
    config = model.config
    if hasattr(config, "n_embd"):
        return config.n_embd
    elif hasattr(config, "hidden_size"):
        return config.hidden_size
    else:
        raise ValueError(f"Cannot determine hidden_size from config: {config}")


def get_layer_module(model, layer_idx):
    """
    获取指定层的 transformer module，用于 register_forward_hook。

    兼容 GPT-2 (model.transformer.h[i]) 和 Llama (model.model.layers[i]) 等架构。
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    else:
        raise ValueError(
            f"Unsupported model architecture: {type(model).__name__}. "
            f"Cannot find transformer layers. "
            f"Available attributes: {dir(model)}"
        )
