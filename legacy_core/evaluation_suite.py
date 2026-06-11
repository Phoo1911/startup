# core/evaluation_suite.py
"""
논문 연구용 종합 평가 Suite
- 추천 품질 평가 (Precision, Recall, F1, NDCG, MRR)
- RAG 품질 평가 (Answer Relevance, Faithfulness)
- Agentic AI 평가 (Tool Usage, Planning Efficiency)
- 비교 실험 (Baseline vs Agentic)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json
from pathlib import Path


# ═══════════════════════════════════════════════
# 📊 평가 결과 데이터 클래스
# ═══════════════════════════════════════════════

@dataclass
class EvaluationMetrics:
    """평가 지표 모음"""
    
    # 추천 품질
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    
    # RAG 품질
    answer_relevance: float = 0.0
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    # Agentic AI
    tool_usage_efficiency: float = 0.0
    planning_quality: float = 0.0
    avg_steps: float = 0.0
    
    # 기타
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "recommendation": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1_score,
                "ndcg": self.ndcg,
                "mrr": self.mrr
            },
            "rag": {
                "answer_relevance": self.answer_relevance,
                "faithfulness": self.faithfulness,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall
            },
            "agentic": {
                "tool_usage_efficiency": self.tool_usage_efficiency,
                "planning_quality": self.planning_quality,
                "avg_steps": self.avg_steps
            },
            "performance": {
                "latency_ms": self.latency_ms
            }
        }


# ═══════════════════════════════════════════════
# 🎯 추천 품질 평가
# ═══════════════════════════════════════════════

class RecommendationEvaluator:
    """추천 시스템 평가"""
    
    @staticmethod
    def precision_at_k(predicted: List[str], ground_truth: List[str], k: int = 10) -> float:
        """Precision@K"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        hits = len(set(predicted_k) & set(ground_truth))
        return hits / min(k, len(predicted_k))
    
    @staticmethod
    def recall_at_k(predicted: List[str], ground_truth: List[str], k: int = 10) -> float:
        """Recall@K"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        hits = len(set(predicted_k) & set(ground_truth))
        return hits / len(ground_truth)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """F1 Score"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(predicted: List[str], ground_truth: List[str], k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG@K)
        추천 순서의 품질을 평가
        """
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        
        # DCG 계산
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            if item in ground_truth:
                relevance = 1
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1)=0
        
        # IDCG 계산 (이상적인 순서)
        ideal_k = min(len(ground_truth), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_k))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mrr(predicted: List[str], ground_truth: List[str]) -> float:
        """
        Mean Reciprocal Rank (MRR)
        첫 번째 정답의 순위로 평가
        """
        if not predicted or not ground_truth:
            return 0.0
        
        for i, item in enumerate(predicted):
            if item in ground_truth:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_single(
        self, 
        predicted_ids: List[str], 
        ground_truth_ids: List[str],
        k: int = 10
    ) -> Dict[str, float]:
        """단일 쿼리 평가"""
        
        precision = self.precision_at_k(predicted_ids, ground_truth_ids, k)
        recall = self.recall_at_k(predicted_ids, ground_truth_ids, k)
        f1 = self.f1_score(precision, recall)
        ndcg = self.ndcg_at_k(predicted_ids, ground_truth_ids, k)
        mrr_score = self.mrr(predicted_ids, ground_truth_ids)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ndcg": ndcg,
            "mrr": mrr_score
        }
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, Any]:
        """
        배치 평가
        
        test_cases = [
            {
                "query_id": "Q1",
                "predicted": ["DOC1", "DOC2", ...],
                "ground_truth": ["DOC1", "DOC5", ...]
            },
            ...
        ]
        """
        results = []
        
        for case in test_cases:
            query_id = case.get("query_id")
            predicted = case.get("predicted", [])
            ground_truth = case.get("ground_truth", [])
            
            metrics = self.evaluate_single(predicted, ground_truth, k)
            metrics["query_id"] = query_id
            results.append(metrics)
        
        # 평균 계산
        avg_metrics = {
            "precision": np.mean([r["precision"] for r in results]),
            "recall": np.mean([r["recall"] for r in results]),
            "f1": np.mean([r["f1"] for r in results]),
            "ndcg": np.mean([r["ndcg"] for r in results]),
            "mrr": np.mean([r["mrr"] for r in results])
        }
        
        return {
            "average": avg_metrics,
            "per_query": results,
            "total_queries": len(test_cases)
        }


# ═══════════════════════════════════════════════
# 📚 RAG 품질 평가
# ═══════════════════════════════════════════════

class RAGQualityEvaluator:
    """RAG 시스템 품질 평가"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def answer_relevance(
        self, 
        question: str, 
        answer: str,
        method: str = "llm"
    ) -> float:
        """
        Answer Relevance: 답변이 질문에 얼마나 관련있는가?
        
        method:
            - "llm": LLM으로 평가 (0~1)
            - "keyword": 키워드 오버랩 (간단)
        """
        if method == "llm" and self.llm:
            prompt = f"""
다음 질문과 답변이 얼마나 관련있는지 0~1 사이 점수로 평가하세요.

질문: {question}
답변: {answer}

0.0 = 전혀 관련없음
0.5 = 부분적으로 관련있음
1.0 = 완벽하게 관련있음

점수만 출력하세요 (예: 0.8):
"""
            try:
                response = self.llm.generate(prompt, max_tokens=10)
                score_str = str(response).strip()
                score = float(score_str)
                return max(0.0, min(1.0, score))
            except:
                pass
        
        # 폴백: 키워드 매칭
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        overlap = len(q_words & a_words)
        return overlap / max(len(q_words), 1)
    
    def faithfulness(
        self,
        answer: str,
        contexts: List[str],
        method: str = "llm"
    ) -> float:
        """
        Faithfulness: 답변이 검색된 컨텍스트에 충실한가?
        (환각 방지)
        """
        if not contexts:
            return 0.0
        
        contexts_text = "\n\n".join(contexts[:3])
        
        if method == "llm" and self.llm:
            prompt = f"""
다음 답변이 제공된 컨텍스트에만 기반한 내용인지 평가하세요.

컨텍스트:
{contexts_text}

답변:
{answer}

평가 기준:
- 1.0: 답변의 모든 내용이 컨텍스트에 있음
- 0.5: 일부는 컨텍스트 기반, 일부는 추론/외부 지식
- 0.0: 컨텍스트와 무관한 내용

점수만 출력하세요:
"""
            try:
                response = self.llm.generate(prompt, max_tokens=10)
                score = float(str(response).strip())
                return max(0.0, min(1.0, score))
            except:
                pass
        
        # 폴백: 단순 오버랩
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())
        overlap = len(answer_words & context_words)
        return overlap / max(len(answer_words), 1)
    
    def context_precision(
        self,
        contexts: List[str],
        ground_truth_contexts: List[str]
    ) -> float:
        """검색된 컨텍스트가 정답 컨텍스트와 얼마나 일치하는가?"""
        if not contexts or not ground_truth_contexts:
            return 0.0
        
        hits = 0
        for ctx in contexts[:5]:
            for gt_ctx in ground_truth_contexts:
                # 간단한 문자열 포함 체크
                if ctx in gt_ctx or gt_ctx in ctx:
                    hits += 1
                    break
        
        return hits / min(5, len(contexts))
    
    def context_recall(
        self,
        contexts: List[str],
        ground_truth_contexts: List[str]
    ) -> float:
        """정답 컨텍스트가 검색 결과에 얼마나 포함되었는가?"""
        if not contexts or not ground_truth_contexts:
            return 0.0
        
        hits = 0
        for gt_ctx in ground_truth_contexts:
            for ctx in contexts:
                if gt_ctx in ctx or ctx in gt_ctx:
                    hits += 1
                    break
        
        return hits / len(ground_truth_contexts)


# ═══════════════════════════════════════════════
# 🤖 Agentic AI 평가
# ═══════════════════════════════════════════════

class AgenticAIEvaluator:
    """Agentic AI 평가"""
    
    @staticmethod
    def tool_usage_efficiency(agent_steps: List[Dict]) -> float:
        """
        도구 사용 효율성
        = 성공한 도구 호출 / 전체 도구 호출
        """
        if not agent_steps:
            return 0.0
        
        total_calls = len(agent_steps)
        successful_calls = sum(
            1 for step in agent_steps 
            if "error" not in str(step.get("observation", "")).lower()
        )
        
        return successful_calls / total_calls
    
    @staticmethod
    def planning_quality(agent_steps: List[Dict], optimal_steps: int = 3) -> float:
        """
        Planning 품질
        = optimal_steps / actual_steps
        (최소 단계로 목표 달성할수록 좋음)
        """
        actual_steps = len(agent_steps)
        if actual_steps == 0:
            return 0.0
        
        return min(1.0, optimal_steps / actual_steps)
    
    @staticmethod
    def goal_achievement(final_result: Dict) -> float:
        """
        목표 달성도
        - 추천이면 recommendations 개수
        - 챗봇이면 답변 생성 여부
        """
        if "recommendations" in final_result:
            recs = final_result.get("recommendations", [])
            return 1.0 if len(recs) > 0 else 0.0
        
        if "answer" in final_result:
            answer = final_result.get("answer", "")
            return 1.0 if len(answer) > 10 else 0.0
        
        return 0.0


# ═══════════════════════════════════════════════
# 🔬 종합 평가 Suite
# ═══════════════════════════════════════════════

class ComprehensiveEvaluationSuite:
    """논문용 종합 평가 Suite"""
    
    def __init__(self, orchestrator, llm_client):
        self.orchestrator = orchestrator
        self.llm = llm_client
        
        self.rec_evaluator = RecommendationEvaluator()
        self.rag_evaluator = RAGQualityEvaluator(llm_client)
        self.agent_evaluator = AgenticAIEvaluator()
    
    def evaluate_recommendation_system(
        self,
        test_cases: List[Dict[str, Any]],
        mode: str = "hybrid"  # "hybrid" or "agentic"
    ) -> Dict[str, Any]:
        """
        추천 시스템 종합 평가
        
        test_cases = [
            {
                "profile": UserProfile(...),
                "ground_truth_ids": ["DOC1", "DOC2", ...],
                "query_id": "Q1"
            },
            ...
        ]
        """
        import time
        
        results = []
        
        for case in test_cases:
            profile = case["profile"]
            ground_truth = case["ground_truth_ids"]
            query_id = case.get("query_id", "")
            
            # 추천 실행
            start_time = time.time()
            
            if mode == "hybrid":
                report = self.orchestrator.run(profile, top_n=10)
            else:  # agentic
                report = self.orchestrator.run_agentic(profile, top_n=10)
            
            latency = (time.time() - start_time) * 1000  # ms
            
            # 예측 결과 추출
            predicted_ids = [
                rec.get("id") 
                for rec in report.get("recommendations", [])
            ]
            
            # 추천 품질 평가
            rec_metrics = self.rec_evaluator.evaluate_single(
                predicted_ids, 
                ground_truth,
                k=10
            )
            
            # Agentic AI 평가 (mode가 agentic인 경우)
            agent_metrics = {}
            if mode == "agentic" and "agent_steps" in report:
                steps = report.get("agent_steps", [])
                agent_metrics = {
                    "tool_efficiency": self.agent_evaluator.tool_usage_efficiency(steps),
                    "planning_quality": self.agent_evaluator.planning_quality(steps, optimal_steps=3),
                    "avg_steps": len(steps)
                }
            
            results.append({
                "query_id": query_id,
                "recommendation_metrics": rec_metrics,
                "agent_metrics": agent_metrics,
                "latency_ms": latency,
                "num_predicted": len(predicted_ids),
                "num_ground_truth": len(ground_truth)
            })
        
        # 평균 계산
        avg_rec = {
            key: np.mean([r["recommendation_metrics"][key] for r in results])
            for key in ["precision", "recall", "f1", "ndcg", "mrr"]
        }
        
        avg_agent = {}
        if mode == "agentic":
            avg_agent = {
                key: np.mean([r["agent_metrics"][key] for r in results if r["agent_metrics"]])
                for key in ["tool_efficiency", "planning_quality", "avg_steps"]
            }
        
        avg_latency = np.mean([r["latency_ms"] for r in results])
        
        return {
            "mode": mode,
            "summary": {
                "recommendation": avg_rec,
                "agentic": avg_agent,
                "latency_ms": avg_latency,
                "total_cases": len(test_cases)
            },
            "per_case": results
        }
    
    def evaluate_rag_quality(
        self,
        qa_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        RAG 품질 평가
        
        qa_pairs = [
            {
                "question": "부산 청년 지원사업은?",
                "answer": "...",
                "contexts": ["...", "..."],
                "ground_truth_contexts": ["...", "..."]
            },
            ...
        ]
        """
        results = []
        
        for pair in qa_pairs:
            question = pair["question"]
            answer = pair["answer"]
            contexts = pair.get("contexts", [])
            gt_contexts = pair.get("ground_truth_contexts", [])
            
            # 평가
            relevance = self.rag_evaluator.answer_relevance(question, answer)
            faithfulness = self.rag_evaluator.faithfulness(answer, contexts)
            ctx_precision = self.rag_evaluator.context_precision(contexts, gt_contexts)
            ctx_recall = self.rag_evaluator.context_recall(contexts, gt_contexts)
            
            results.append({
                "question": question[:50] + "...",
                "answer_relevance": relevance,
                "faithfulness": faithfulness,
                "context_precision": ctx_precision,
                "context_recall": ctx_recall
            })
        
        # 평균
        avg = {
            "answer_relevance": np.mean([r["answer_relevance"] for r in results]),
            "faithfulness": np.mean([r["faithfulness"] for r in results]),
            "context_precision": np.mean([r["context_precision"] for r in results]),
            "context_recall": np.mean([r["context_recall"] for r in results])
        }
        
        return {
            "summary": avg,
            "per_question": results
        }
    
    def compare_methods(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Baseline vs Agentic AI 비교
        """
        print("\n" + "="*80)
        print("🔬 방법론 비교 평가")
        print("="*80)
        
        # Baseline (Hybrid) 평가
        print("\n1️⃣ Baseline (Semantic Matching) 평가 중...")
        baseline_results = self.evaluate_recommendation_system(
            test_cases, 
            mode="hybrid"
        )
        
        # Agentic AI 평가
        print("\n2️⃣ Agentic AI 평가 중...")
        agentic_results = self.evaluate_recommendation_system(
            test_cases,
            mode="agentic"
        )
        
        # 비교 리포트
        comparison = {
            "baseline": baseline_results["summary"],
            "agentic": agentic_results["summary"],
            "improvement": {
                "precision": (
                    agentic_results["summary"]["recommendation"]["precision"] -
                    baseline_results["summary"]["recommendation"]["precision"]
                ),
                "recall": (
                    agentic_results["summary"]["recommendation"]["recall"] -
                    baseline_results["summary"]["recommendation"]["recall"]
                ),
                "f1": (
                    agentic_results["summary"]["recommendation"]["f1"] -
                    baseline_results["summary"]["recommendation"]["f1"]
                ),
                "ndcg": (
                    agentic_results["summary"]["recommendation"]["ndcg"] -
                    baseline_results["summary"]["recommendation"]["ndcg"]
                )
            }
        }
        
        print("\n" + "="*80)
        print("📊 비교 결과")
        print("="*80)
        print(f"\nPrecision: {baseline_results['summary']['recommendation']['precision']:.3f} → {agentic_results['summary']['recommendation']['precision']:.3f}")
        print(f"Recall:    {baseline_results['summary']['recommendation']['recall']:.3f} → {agentic_results['summary']['recommendation']['recall']:.3f}")
        print(f"F1:        {baseline_results['summary']['recommendation']['f1']:.3f} → {agentic_results['summary']['recommendation']['f1']:.3f}")
        print(f"NDCG:      {baseline_results['summary']['recommendation']['ndcg']:.3f} → {agentic_results['summary']['recommendation']['ndcg']:.3f}")
        
        return comparison
    
    def save_results(self, results: Dict, output_path: str):
        """결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {output_path}")