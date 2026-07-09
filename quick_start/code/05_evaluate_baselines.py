"""Evaluate the included SAUP-Multiple quick-start baselines."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve


QUICK_START = Path(__file__).resolve().parents[1]
PAPER_AUROC_EXPANSION = 20


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def accuracy_from_labels(labels) -> float:
    values = list(labels)
    return sum(str(x).lower() == "correct" for x in values) / max(len(values), 1)


def auarc(accuracy: np.ndarray, uncertainty: np.ndarray) -> float:
    order = np.argsort(uncertainty)
    sorted_acc = accuracy[order]
    coverages = np.arange(1, len(sorted_acc) + 1, dtype=float) / len(sorted_acc)
    acc_curve = np.cumsum(sorted_acc) / np.arange(1, len(sorted_acc) + 1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(acc_curve, coverages))
    return float(np.trapz(acc_curve, coverages))


def paper_auroc(accuracies: np.ndarray, certainty_scores: np.ndarray) -> float:
    y_true: list[int] = []
    y_score: list[float] = []
    for accuracy, score in zip(accuracies, certainty_scores):
        correct_runs = int(round(float(accuracy) * PAPER_AUROC_EXPANSION))
        for idx in range(PAPER_AUROC_EXPANSION):
            y_true.append(1 if idx < correct_runs else 0)
            y_score.append(float(score))
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    return float(auc(fpr, tpr))


def evaluate(name: str, scores: dict, labels: dict, *, invert: bool = False) -> None:
    keys = sorted(set(scores) & set(labels))
    uncertainty = []
    certainty = []
    accuracy = []
    for key in keys:
        values = np.asarray(scores[key], dtype=float)
        score = float(np.nanmean(values))
        uncertainty.append(-score if invert else score)
        certainty.append(score if invert else -score)
        accuracy.append(accuracy_from_labels(labels[key]))

    uncertainty_arr = np.asarray(uncertainty, dtype=float)
    certainty_arr = np.asarray(certainty, dtype=float)
    accuracy_arr = np.asarray(accuracy, dtype=float)
    print(f"{name}:")
    print(f"  tasks: {len(keys)}")
    print(f"  AUROC: {paper_auroc(accuracy_arr, certainty_arr):.4f}")
    print(f"  AUARC: {auarc(accuracy_arr, uncertainty_arr):.4f}")


def evaluate_math_qwen() -> None:
    labels = load_pickle(QUICK_START / "results" / "accuracy_dict_Math_qwen2.5.pkl")
    saup = load_pickle(QUICK_START / "results" / "saup_scores_Math_qwen2.5.pkl")
    evaluate("MATH + Qwen2.5 + SAUP-Multiple", saup, labels, invert=True)


def evaluate_mmlu_autogen_qwen() -> None:
    labels = load_pickle(QUICK_START / "results" / "accuracy_dict_MMLU_Autogen_qwen2.5.pkl")
    saup = load_pickle(QUICK_START / "results" / "saup_scores_MMLU_Autogen_qwen2.5.pkl")
    evaluate("MMLU + AutoGen + Qwen2.5 + SAUP-Multiple", saup, labels, invert=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate included SAUP-Multiple quick-start baselines.")
    parser.add_argument(
        "--sample",
        choices=["math-qwen", "mmlu-autogen-qwen", "all"],
        default="all",
        help="Baseline sample to evaluate.",
    )
    args = parser.parse_args()

    if args.sample in {"math-qwen", "all"}:
        evaluate_math_qwen()
    if args.sample == "all":
        print()
    if args.sample in {"mmlu-autogen-qwen", "all"}:
        evaluate_mmlu_autogen_qwen()


if __name__ == "__main__":
    main()
