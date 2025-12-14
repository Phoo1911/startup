# tests/run_eval_exp001.py
from core.evaluator import RAGEvaluator
from core.rag_system import RAGSystem
from core.llm_client import LLMClient
import json, pathlib

def load_eval_data(path: str):
    # JSON/CSV 에서 평가용 질의/정답 로드
    ...

if __name__ == "__main__":
    rag = RAGSystem(...)
    llm = LLMClient(...)
    evaluator = RAGEvaluator(rag, llm)

    samples = load_eval_data("reports/datasets/eval_rag_v1.json")
    result = evaluator.evaluate_batch(samples)

    out_path = pathlib.Path("reports/results/eval_rag_exp001.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("✔ saved to", out_path)
