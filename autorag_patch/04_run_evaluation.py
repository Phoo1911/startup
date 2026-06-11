from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


MODE_DEFAULTS: Dict[str, Tuple[str, str]] = {
    "baseline": ("autorag_workspace/configs/benchmark_1.yaml", "autorag_workspace/results_baseline"),
    "autorag": ("autorag_workspace/configs/benchmark_3.yaml", "autorag_workspace/results_autorag"),
    "full": ("autorag_workspace/configs/rag_eval.yaml", "autorag_workspace/results_full"),
    "full_bgem3_top5": (
        "autorag_workspace/configs/rag_eval_bgem3_top5.yaml",
        "autorag_workspace/results_full_bgem3_top5",
    ),
    "backbone_fresh": (
        "autorag_workspace/configs/benchmark_fresh_all.yaml",
        "autorag_workspace/results_backbone_fresh",
    ),
    "retrieval_strategy_fresh": (
        "autorag_workspace/configs/retrieval_strategy_fresh.yaml",
        "autorag_workspace/results_retrieval_strategy_fresh",
    ),
}


def _resolve_mode_paths(mode: str, config_path: str | None, project_dir: str | None) -> tuple[str, str]:
    key = str(mode or "full").strip().lower()
    if key not in MODE_DEFAULTS:
        allowed = " | ".join(MODE_DEFAULTS.keys())
        raise ValueError(f"Unsupported mode: {mode!r}. Use {allowed}")

    default_config, default_project = MODE_DEFAULTS[key]
    return config_path or default_config, project_dir or default_project


def _resolve_corpus_data_path(corpus_data_path: str | None) -> str:
    if corpus_data_path:
        return corpus_data_path

    corpus_root = Path("autorag_workspace/corpus")
    candidates = [p for p in corpus_root.rglob("*.parquet") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No corpus parquet found under: {corpus_root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def run_autorag_evaluation(
    mode: str = "full",
    config_path: str | None = None,
    qa_data_path: str = "autorag_workspace/qa/qa.parquet",
    corpus_data_path: str | None = None,
    project_dir: str | None = None,
    stream_output: bool = True,
) -> None:
    config_path, project_dir = _resolve_mode_paths(mode, config_path, project_dir)
    corpus_data_path = _resolve_corpus_data_path(corpus_data_path)

    print("RAG ?? ??...")
    print(f"   - mode: {mode}")
    print(f"   - config: {config_path}")
    print(f"   - project_dir: {project_dir}")

    cmd = [
        "autorag",
        "evaluate",
        "--config",
        config_path,
        "--qa_data_path",
        qa_data_path,
        "--corpus_data_path",
        corpus_data_path,
        "--project_dir",
        project_dir,
    ]

    if stream_output:
        try:
            subprocess.run(cmd, check=True)
            print(f"\n?? ??: {project_dir}")
        except subprocess.CalledProcessError as exc:
            print(f"?? ??: {exc}")
            raise
        return

    try:
        subprocess.run(cmd, check=True)
        print(f"\n?? ??: {project_dir}")
    except subprocess.CalledProcessError as exc:
        print(f"?? ??: {exc}")
        if exc.stdout:
            print("[autorag stdout]")
            print(exc.stdout)
        if exc.stderr:
            print("[autorag stderr]")
            print(exc.stderr)
        raise


def run_hybrid_postprocess(
    project_dir: str,
    qa_data_path: str,
    dense_module_pattern: str = "vectordb_bgem3",
) -> None:
    print("\nbenchmark ??? hybrid retrieval(RRF) ?? ?...")
    cmd = [
        sys.executable,
        "autorag_patch/06_compute_hybrid_from_benchmark.py",
        "--project-dir",
        project_dir,
        "--qa-path",
        qa_data_path,
        "--dense-module-pattern",
        dense_module_pattern,
    ]
    subprocess.run(cmd, check=True)


def _latest_trial_dir(project_dir: Path) -> Path | None:
    if not project_dir.exists():
        return None
    dirs = [p for p in project_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: int(p.name))


def view_dashboard(project_dir: str = "autorag_workspace/results") -> None:
    project_dir_path = Path(project_dir)
    trial_dir = _latest_trial_dir(project_dir_path)
    if trial_dir is None:
        print(f"trial_dir? ?? ?????: {project_dir}")
        return

    print(f"???? ??: {trial_dir}")
    cmd = ["autorag", "dashboard", "--trial_dir", str(trial_dir)]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoRAG evaluation by experiment mode")
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "autorag",
            "full",
            "full_bgem3_top5",
            "backbone_fresh",
            "retrieval_strategy_fresh",
        ],
        default="full",
    )
    parser.add_argument("--config-path", default=None, help="Override config path")
    parser.add_argument("--qa-data-path", default="autorag_workspace/qa/qa.parquet")
    parser.add_argument("--corpus-data-path", default=None)
    parser.add_argument("--project-dir", default=None, help="Override output project dir")
    parser.add_argument("--capture-output", action="store_true", help="Capture stdout/stderr instead of streaming")
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()

    run_autorag_evaluation(
        mode=args.mode,
        config_path=args.config_path,
        qa_data_path=args.qa_data_path,
        corpus_data_path=args.corpus_data_path,
        project_dir=args.project_dir,
        stream_output=not args.capture_output,
    )
    if args.mode == "retrieval_strategy_fresh":
        _, resolved_project_dir = _resolve_mode_paths(args.mode, args.config_path, args.project_dir)
        run_hybrid_postprocess(
            project_dir=resolved_project_dir,
            qa_data_path=args.qa_data_path,
        )
    if not args.no_dashboard:
        _, resolved_project_dir = _resolve_mode_paths(args.mode, args.config_path, args.project_dir)
        view_dashboard(project_dir=resolved_project_dir)

