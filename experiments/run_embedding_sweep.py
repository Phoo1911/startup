from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "autorag_workspace" / "results_agent"
DEFAULT_SWEEP_DIR = PROJECT_ROOT / "autorag_workspace" / "experiment_results" / "embedding_sweep"
DEFAULT_QA_PATH = PROJECT_ROOT / "autorag_workspace" / "qa" / "qa.parquet"
DEFAULT_MODELS = (
    "dragonkue/BGE-m3-ko",
    "intfloat/multilingual-e5-large-instruct",
    "BM-K/KoSimCSE-roberta",
)
DEFAULT_JUDGE_BACKEND = "transformers"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-8B"


def _safe_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _load_metric_row(csv_path: Path, mode: str = "ALL") -> dict[str, object]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "mode" in df.columns:
        subset = df[df["mode"].astype(str) == mode]
        if subset.empty:
            subset = df.iloc[:1]
    else:
        subset = df.iloc[:1]
    return subset.iloc[0].to_dict() if len(subset) else {}


def _mean_latency(predictions_path: Path) -> float | None:
    if not predictions_path.exists():
        return None
    df = pd.read_parquet(predictions_path, columns=["latency"])
    if "latency" not in df.columns or df.empty:
        return None
    return float(df["latency"].dropna().mean()) if df["latency"].notna().any() else None


def _collect_summary_row(model_name: str, run_dir: Path) -> dict[str, object]:
    predictions_path = run_dir / "predictions.parquet"
    policy_path = run_dir / "policy_metrics.csv"
    hallucination_path = run_dir / "hallucination_metrics.csv"
    policy = _load_metric_row(policy_path, mode="ALL")
    halluc = _load_metric_row(hallucination_path, mode="ALL")
    return {
        "embedding_model": model_name,
        "precision": policy.get("precision"),
        "recall": policy.get("recall"),
        "f1_score": policy.get("f1_score"),
        "top1_accuracy": policy.get("top1_accuracy"),
        "n_samples": policy.get("n_samples"),
        "hallucination_rate": halluc.get("hallucination_rate"),
        "grounded_ratio": halluc.get("grounded_ratio"),
        "unknown_ratio": halluc.get("unknown_ratio"),
        "correctness_avg": halluc.get("correctness_avg"),
        "faithfulness_avg": halluc.get("faithfulness_avg"),
        "avg_latency_sec": _mean_latency(predictions_path),
        "run_dir": str(run_dir),
    }


def run_sweep(
    models: Iterable[str],
    qa_path: Path,
    sweep_dir: Path,
    judge_backend: str,
    judge_model: str,
    modes: list[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")

    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for model_name in models:
        print(f"\n=== Embedding Sweep: {model_name} ===")
        env = os.environ.copy()
        env["AH_EMBEDDING_MODEL"] = model_name
        env["AH_VECTORSTORE_DIR"] = f"cache_{_safe_name(model_name)}"

        run_dir = sweep_dir / _safe_name(model_name)
        run_dir.mkdir(parents=True, exist_ok=True)

        rebuild_cmd = [
            sys.executable,
            "-c",
            (
                "from dotenv import load_dotenv; "
                "load_dotenv('.env'); "
                "from config.settings import Config; "
                "from legacy_core.orchestrator import AgenticOrchestrator; "
                "orch = AgenticOrchestrator(service_key=Config.SERVICE_KEY, llm_api_key=Config.LLM_API_KEY); "
                "orch.ensure_index(use_cache=False); "
                "print('done: rebuilt cache for embedding sweep')"
            ),
        ]
        _run(rebuild_cmd, env)

        predictions_path = results_dir / "predictions.parquet"
        policy_path = results_dir / "policy_metrics.csv"
        hallucination_path = results_dir / "hallucination_metrics.csv"

        eval_cmd = [sys.executable, "run_agent_eval.py", "--qa-path", str(qa_path)]
        if modes:
            eval_cmd.extend(["--modes", *modes])
        if limit is not None:
            eval_cmd.extend(["--limit", str(limit)])
        _run(eval_cmd, env)

        _run(
            [
                sys.executable,
                "evaluate_policy_accuracy.py",
                "--input-path",
                str(predictions_path),
                "--output-path",
                str(policy_path),
            ],
            env,
        )

        _run(
            [
                sys.executable,
                "evaluate_hallucination.py",
                "--input-path",
                str(predictions_path),
                "--output-path",
                str(hallucination_path),
                "--judge-backend",
                judge_backend,
                "--judge-model",
                judge_model,
            ],
            env,
        )

        _copy_if_exists(predictions_path, run_dir / "predictions.parquet")
        _copy_if_exists(policy_path, run_dir / "policy_metrics.csv")
        _copy_if_exists(hallucination_path, run_dir / "hallucination_metrics.csv")
        summary_rows.append(_collect_summary_row(model_name, run_dir))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = sweep_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] summary saved to {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare embedding models with rebuilt indexes")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--judge-backend", type=str, default=DEFAULT_JUDGE_BACKEND)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--modes", nargs="+", default=["full"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_sweep(
        models=args.models,
        qa_path=args.qa_path,
        sweep_dir=args.sweep_dir,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        modes=args.modes,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
