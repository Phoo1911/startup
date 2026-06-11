from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


QA_PATH = Path("autorag_workspace/qa/qa.parquet")
CORPUS_PATH = Path("autorag_workspace/qa/corpus.parquet")
AUTORAG_CONFIG_PATH = Path("autorag_workspace/configs/rag_eval.yaml")
AUTORAG_PROJECT_DIR = Path("autorag_workspace/results_experiment_autorag")

PREDICTIONS_PATH = Path("autorag_workspace/results_agent/predictions.parquet")
POLICY_METRICS_PATH = Path("autorag_workspace/results_agent/policy_metrics.csv")
HALLUCINATION_METRICS_PATH = Path("autorag_workspace/results_agent/hallucination_metrics.csv")
FINAL_SUMMARY_PATH = Path("autorag_workspace/experiment_results/summary.csv")

DEFAULT_MODES = [
    "baseline",
    "autorag",
    "full",
    "no_hyde",
    "no_reranker",
    "no_freshness",
    "no_deadline",
]
DEFAULT_JUDGE_BACKEND = "openai"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-8B"


def _run_cmd(cmd: List[str], step_name: str) -> None:
    print(f"\n=== {step_name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_autorag_once() -> None:
    cmd = [
        "autorag",
        "evaluate",
        "--config",
        str(AUTORAG_CONFIG_PATH),
        "--qa_data_path",
        str(QA_PATH),
        "--corpus_data_path",
        str(CORPUS_PATH),
        "--project_dir",
        str(AUTORAG_PROJECT_DIR),
    ]
    _run_cmd(cmd, "Step 1/5 AutoRAG evaluate (once)")


def _run_agent_batch(modes: List[str], limit: int | None) -> None:
    cmd = [
        sys.executable,
        "run_agent_eval.py",
        "--qa-path",
        str(QA_PATH),
        "--output-path",
        str(PREDICTIONS_PATH),
        "--modes",
        *modes,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    _run_cmd(cmd, "Step 2/5 Agent batch evaluation")


def _run_policy_accuracy() -> None:
    cmd = [
        sys.executable,
        "evaluate_policy_accuracy.py",
        "--input-path",
        str(PREDICTIONS_PATH),
        "--output-path",
        str(POLICY_METRICS_PATH),
    ]
    _run_cmd(cmd, "Step 3/5 Policy accuracy")


def _run_hallucination(judge_backend: str, judge_model: str, max_new_tokens: int) -> None:
    cmd = [
        sys.executable,
        "evaluate_hallucination.py",
        "--input-path",
        str(PREDICTIONS_PATH),
        "--output-path",
        str(HALLUCINATION_METRICS_PATH),
        "--judge-backend",
        judge_backend,
        "--judge-model",
        judge_model,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    _run_cmd(cmd, "Step 4/5 Hallucination evaluation")


def _build_final_summary(target_modes: List[str]) -> pd.DataFrame:
    print("\n=== Step 5/5 Build final summary ===")
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDICTIONS_PATH}")
    if not POLICY_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing policy metrics: {POLICY_METRICS_PATH}")
    if not HALLUCINATION_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing hallucination metrics: {HALLUCINATION_METRICS_PATH}")

    pred_df = pd.read_parquet(PREDICTIONS_PATH)
    policy_df = pd.read_csv(POLICY_METRICS_PATH)
    hall_df = pd.read_csv(HALLUCINATION_METRICS_PATH)

    latency_df = (
        pred_df.groupby("mode", dropna=False)["latency"]
        .mean()
        .reset_index()
        .rename(columns={"latency": "average_latency"})
    )

    policy_df = policy_df.rename(
        columns={
            "precision": "policy_precision",
            "recall": "policy_recall",
            "f1_score": "policy_f1",
        }
    )
    policy_df = policy_df[["mode", "policy_precision", "policy_recall", "policy_f1"]]
    hall_df = hall_df[["mode", "hallucination_rate", "correctness_avg", "faithfulness_avg"]]

    merged = latency_df.merge(policy_df, on="mode", how="left").merge(hall_df, on="mode", how="left")

    # Keep requested comparison modes first, then any extra.
    ordered = [m for m in target_modes if m in set(merged["mode"].astype(str))]
    extras = [m for m in merged["mode"].astype(str).tolist() if m not in ordered and m != "ALL"]
    final_modes = ordered + extras
    merged["mode"] = merged["mode"].astype(str)
    merged = merged[merged["mode"].isin(final_modes)]
    merged["mode"] = pd.Categorical(merged["mode"], categories=final_modes, ordered=True)
    merged = merged.sort_values("mode").reset_index(drop=True)

    out_df = merged[
        [
            "mode",
            "policy_precision",
            "policy_recall",
            "policy_f1",
            "hallucination_rate",
            "correctness_avg",
            "faithfulness_avg",
            "average_latency",
        ]
    ]
    FINAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(FINAL_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(out_df.to_string(index=False))
    print(f"\nSaved: {FINAL_SUMMARY_PATH}")
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--limit", type=int, default=None, help="Optional max questions for quick runs")
    parser.add_argument("--judge-backend", choices=["openai", "transformers"], default=DEFAULT_JUDGE_BACKEND)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-autorag", action="store_true", help="Skip step 1 autorag evaluate")
    args = parser.parse_args()

    modes = [str(m).strip().lower() for m in args.modes if str(m).strip()]
    allowed = set(DEFAULT_MODES)
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Allowed: {sorted(allowed)}")

    if not args.skip_autorag:
        _run_autorag_once()
    _run_agent_batch(modes=modes, limit=args.limit)
    _run_policy_accuracy()
    _run_hallucination(
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        max_new_tokens=args.judge_max_new_tokens,
    )
    _build_final_summary(target_modes=modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
