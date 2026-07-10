"""
线性探针模块：对每层隐藏状态训练 logistic regression 二分类器（技能 vs 知识）。

输出：
  - results/{run_id}/probe_results.json  — 每层探针准确率、权重
  - results/{run_id}/probe_weights.json  — 探针权重向量（用于消融实验）
"""

import argparse
import json
import os

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_hidden_states(hidden_states_path):
    """加载提取的隐藏状态。"""
    with open(hidden_states_path, "r") as f:
        data = json.load(f)
    return data


def prepare_probe_data(hidden_data, layer_idx):
    """
    从隐藏状态数据中提取指定层的特征矩阵和标签。

    返回:
        X: ndarray (num_samples, hidden_size)
        y: ndarray (num_samples,) — 1=skill, 0=knowledge
        sample_ids: list
    """
    results = hidden_data["results"]
    X = []
    y = []
    sample_ids = []

    for r in results:
        hidden = r["hidden_states"][str(layer_idx)]
        X.append(hidden)
        y.append(r["label"])
        sample_ids.append(r["id"])

    X = np.array(X)
    y = np.array(y)
    return X, y, sample_ids


def train_probe(X, y, cfg_probe):
    """
    训练线性探针并进行评估。

    使用 Pipeline(StandardScaler + LogisticRegression) 避免数据泄露：
    - StandardScaler 只在训练集上 fit，然后 transform 验证/测试集
    - CV 中每个 fold 内独立 fit scaler

    返回:
        result: dict 包含各指标
    """
    # 先做 train/test split（在原始空间中）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg_probe.get("test_size", 0.2),
        random_state=cfg_probe.get("random_state", 42),
        stratify=y,
    )

    # 5-fold cross validation on training set, 每个 fold 内独立做 scaler fit
    cv = StratifiedKFold(n_splits=cfg_probe.get("cv_folds", 5), shuffle=True,
                         random_state=cfg_probe.get("random_state", 42))
    cv_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Pipeline 确保 scaler 只在 fold 的训练集上 fit
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=cfg_probe.get("C", 1.0),
                max_iter=cfg_probe.get("max_iter", 1000),
                random_state=cfg_probe.get("random_state", 42),
            )),
        ])
        pipe.fit(X_fold_train, y_fold_train)
        acc = pipe.score(X_fold_val, y_fold_val)
        cv_accuracies.append(acc)

    # 在完整训练集上训练最终模型（Pipeline 确保 scaler 只 fit 训练集）
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=cfg_probe.get("C", 1.0),
            max_iter=cfg_probe.get("max_iter", 1000),
            random_state=cfg_probe.get("random_state", 42),
        )),
    ])
    final_pipe.fit(X_train, y_train)

    test_accuracy = final_pipe.score(X_test, y_test)

    # 获取权重向量
    # Pipeline 中: scaler 是第一步, clf 是第二步
    # 权重向量在标准化空间中: clf.coef_[0]
    # 要在原始空间使用，需要转换: w_original = w_scaled / scale
    # 这个转换在 ablation.py 中完成
    weight_vector = final_pipe.named_steps["clf"].coef_[0]

    result = {
        "cv_mean_accuracy": float(np.mean(cv_accuracies)),
        "cv_std_accuracy": float(np.std(cv_accuracies)),
        "cv_fold_accuracies": [float(a) for a in cv_accuracies],
        "test_accuracy": float(test_accuracy),
        "weight_vector": weight_vector.tolist(),
        "intercept": float(final_pipe.named_steps["clf"].intercept_[0]),
        "scaler_mean": final_pipe.named_steps["scaler"].mean_.tolist(),
        "scaler_scale": final_pipe.named_steps["scaler"].scale_.tolist(),
        "num_train": len(X_train),
        "num_test": len(X_test),
    }

    return result


def run_all_probes(hidden_data, cfg_probe):
    """
    对每层训练探针，返回结果。

    返回:
        layer_results: dict, key=layer_idx, value=probe result dict
        best_layer: int, 探针准确率最高的层
    """
    num_layers = hidden_data["num_layers"]
    hidden_size = hidden_data["hidden_size"]

    layer_results = {}
    best_accuracy = -1
    best_layer = 0

    print(f"[Probe] Training probes for {num_layers} layers...")
    for layer_idx in range(num_layers):
        X, y, sample_ids = prepare_probe_data(hidden_data, layer_idx)

        result = train_probe(X, y, cfg_probe)
        layer_results[str(layer_idx)] = result

        acc = result["cv_mean_accuracy"]
        print(f"  Layer {layer_idx}: CV accuracy = {acc:.4f} +/- {result['cv_std_accuracy']:.4f}, "
              f"Test accuracy = {result['test_accuracy']:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_layer = layer_idx

    print(f"\n[Probe] Best layer: {best_layer} (CV accuracy = {best_accuracy:.4f})")
    return layer_results, best_layer


def check_abort_criteria(layer_results, cfg_abort):
    """检查实验0的放弃条件。"""
    max_accuracy = max(r["cv_mean_accuracy"] for r in layer_results.values())
    threshold = cfg_abort.get("exp0_min_accuracy", 0.60)

    if max_accuracy < threshold:
        print(f"\n[ABORT] Experiment 0 failed: best probe accuracy ({max_accuracy:.4f}) < threshold ({threshold})")
        print("[ABORT] All layers have accuracy < 60%. The skill/knowledge separation hypothesis is not supported by this probe.")
        return True
    else:
        print(f"\n[OK] Experiment 0 passed: best probe accuracy ({max_accuracy:.4f}) >= threshold ({threshold})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate linear probes")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument("--hidden-states", type=str, default=None, help="Path to hidden_states.json (auto-detected if omitted)")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_probe = config["probe"]
    cfg_abort = config.get("abort_criteria", {})

    # 确定路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, config["output"]["results_dir"], args.run_id)

    hidden_states_path = args.hidden_states or os.path.join(results_dir, "hidden_states.json")
    print(f"[Load] Loading hidden states from {hidden_states_path}")
    hidden_data = load_hidden_states(hidden_states_path)

    # 训练探针
    layer_results, best_layer = run_all_probes(hidden_data, cfg_probe)

    # 保存探针结果
    probe_results_path = os.path.join(results_dir, "probe_results.json")
    probe_summary = {
        "model": hidden_data["model"],
        "num_layers": hidden_data["num_layers"],
        "hidden_size": hidden_data["hidden_size"],
        "best_layer": best_layer,
        "best_cv_accuracy": layer_results[str(best_layer)]["cv_mean_accuracy"],
        "layer_results": {k: {kk: vv for kk, vv in v.items() if kk != "weight_vector"}
                         for k, v in layer_results.items()},
    }
    with open(probe_results_path, "w") as f:
        json.dump(probe_summary, f, indent=2)
    print(f"[Save] Probe results saved to {probe_results_path}")

    # 单独保存权重（用于消融实验）
    probe_weights_path = os.path.join(results_dir, "probe_weights.json")
    weights_data = {
        "model": hidden_data["model"],
        "num_layers": hidden_data["num_layers"],
        "hidden_size": hidden_data["hidden_size"],
        "best_layer": best_layer,
        "layer_weights": {},
    }
    for layer_idx, result in layer_results.items():
        weights_data["layer_weights"][layer_idx] = {
            "weight_vector": result["weight_vector"],
            "intercept": result["intercept"],
            "scaler_mean": result["scaler_mean"],
            "scaler_scale": result["scaler_scale"],
        }
    with open(probe_weights_path, "w") as f:
        json.dump(weights_data, f)
    print(f"[Save] Probe weights saved to {probe_weights_path}")

    # 检查放弃条件
    should_abort = check_abort_criteria(layer_results, cfg_abort)
    if should_abort:
        # 记录放弃决定
        abort_path = os.path.join(results_dir, "abort_exp0.json")
        with open(abort_path, "w") as f:
            json.dump({
                "aborted": True,
                "reason": "All layers probe accuracy < 60%",
                "best_accuracy": layer_results[str(best_layer)]["cv_mean_accuracy"],
                "threshold": cfg_abort.get("exp0_min_accuracy", 0.60),
            }, f, indent=2)

    return should_abort


if __name__ == "__main__":
    main()
