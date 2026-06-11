"""AutoRAG 전체 파이프라인을 한 번에 실행.

흐름
1) setup_autorag.py        : 워크스페이스 폴더 생성
2) export_to_autorag.py    : 수집데이터 -> parsed.parquet
3) 02_chunk_corpus.py      : parsed -> corpus(청크)
4) patch_corpus_datatype.py: corpus에 data_type 복구(필수)
5) 03_generate_qa.py        : QA 생성
6) 04_run_evaluation.py     : autorag evaluate + dashboard

중요
- chunker 결과 파일명이 corpus.parquet가 아닐 수 있음(예: corpus/0.parquet)
- 03_generate_qa.py는 --corpus-path를 안 주면 자동으로 '최신 corpus parquet'를 찾음
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTORAG_PATCH_DIR = PROJECT_ROOT / "autorag_patch"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config


def _load_script_module(script_name: str):
    script_path = AUTORAG_PATCH_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"스크립트를 찾을 수 없습니다: {script_path}")

    module_name = f"_autorag_{script_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if not spec or not spec.loader:
        raise ImportError(f"{script_name} 모듈을 로드할 수 없습니다.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(mode: str = "full") -> None:
    print("=" * 80)
    print("🚀 AutoRAG 평가 데이터 생성 파이프라인")
    print("=" * 80)

    try:
        print("\n[1/6] 디렉토리 설정...")
        setup_module = _load_script_module("setup_autorag.py")
        setup_module.setup_autorag_dirs()

        print("\n[2/6] 데이터 변환 중...")
        export_module = _load_script_module("export_to_autorag.py")
        export_module.export_to_parquet()

        print("\n[3/6] 문서 청킹 중...")
        chunk_module = _load_script_module("02_chunk_corpus.py")
        corpus_path = chunk_module.chunk_documents()

        print("\n[4/6] corpus data_type 복구 중(필수)...")
        patch_module = _load_script_module("patch_corpus_datatype.py")
        if isinstance(corpus_path, Path) and corpus_path.is_dir():
            raise FileNotFoundError(
                f"Chunking completed but no corpus parquet was found under: {corpus_path}"
            )

        if hasattr(patch_module, "patch_corpus_datatype"):
            patch_module.patch_corpus_datatype(
                parsed_path=Path("autorag_workspace/parsed/parsed.parquet"),
                corpus_path=Path(corpus_path),
                out_path=None,
                inplace=True,
            )
        elif hasattr(patch_module, "patch_corpus"):
            patch_module.patch_corpus(
                parsed_path=Path("autorag_workspace/parsed/parsed.parquet"),
                corpus_path=Path(corpus_path),
                out_path=None,
                inplace=True,
            )
        else:
            print("⚠️ patch_corpus_datatype.py에 patch_corpus 함수가 없습니다.")
            print("   아래를 직접 실행해 주세요:")
            print("   python autorag_patch/patch_corpus_datatype.py --inplace")

        print("\n[5/6] QA 데이터 생성 중...")
        qa_module = _load_script_module("03_generate_qa.py")
        qa_provider = "openai" if getattr(Config, "LLM_API_KEY", None) else "local"
        if qa_provider == "local":
            print("?? OPENAI API key? ?? QA ?? provider? local? ?????.")
        qa_module.generate_qa_dataset(
            samples_per_type=15,
            llm_provider=qa_provider,
            corpus_path=str(corpus_path),
        )

        print("\n[6/6] RAG 평가 실행 중...")
        eval_module = _load_script_module("04_run_evaluation.py")
        try:
            eval_module.run_autorag_evaluation(mode=mode, corpus_data_path=str(corpus_path))
        except Exception as eval_exc:
            print("?? AutoRAG evaluate ??? ?????, parsed/corpus/qa ??? ???????.")
            print(f"   - ??: {type(eval_exc).__name__}: {eval_exc}")
            print("   - ?? ??: python run_agent_eval.py --limit 20 --modes full")
            print("   - ?? ??: python evaluate_policy_accuracy.py")

        print("\n" + "=" * 80)
        print("✅ 전체 파이프라인 완료!")
        print("=" * 80)
    except Exception as exc:  # pragma: no cover
        print(f"\n❌ 오류 발생: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full AutoRAG data + evaluation pipeline")
    parser.add_argument("--mode", choices=["baseline", "autorag", "full"], default="full")
    args = parser.parse_args()
    main(mode=args.mode)
