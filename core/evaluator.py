# core/evaluator.py
from typing import List, Dict, Any
from statistics import mean
import difflib
import numpy as np

# 선택: BERTScore 사용 (설치 안 되어 있으면 False 처리)
try:
    from bert_score import score as bertscore
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


# ─────────────────────────────────────────────
# 1) 검색 정확도 평가용: RetrievalEvaluator
# ─────────────────────────────────────────────
class RetrievalEvaluator:
    """
    RAG 검색 성능 평가용
    - Precision@K
    - Recall@K
    - nDCG@K
    """

    def __init__(self, rag_system):
        self.rag = rag_system

    def precision_at_k(self, query: str, ground_truth_ids: List[str], k: int = 5) -> float:
        """
        ground_truth_ids: '정답이라고 간주하는 문서 id' 리스트
        """
        results = self.rag.search(query, top_k=k)
        retrieved_ids = [
            getattr(r["document"], "id", None) for r in results
        ]

        hit = sum(1 for x in retrieved_ids if x in ground_truth_ids)
        return hit / max(k, 1)

    def recall_at_k(self, query: str, ground_truth_ids: List[str], k: int = 10) -> float:
        results = self.rag.search(query, top_k=k)
        retrieved_ids = [
            getattr(r["document"], "id", None) for r in results
        ]

        if not ground_truth_ids:
            return 0.0

        hit = sum(1 for x in ground_truth_ids if x in retrieved_ids)
        return hit / len(ground_truth_ids)

    def ndcg_at_k(self, query: str, ground_truth_ids: List[str], k: int = 10) -> float:
        """
        간단한 nDCG 계산 (relevance = 1 또는 0)
        """
        results = self.rag.search(query, top_k=k)
        retrieved_ids = [
            getattr(r["document"], "id", None) for r in results
        ]

        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                dcg += 1.0 / np.log2(i + 2)

        ideal_len = min(len(ground_truth_ids), k)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0




from typing import List, Dict, Any


class MatchingEvaluator:
    """
    추천 적합도 평가용
    - Precision
    - Recall
    - F1-score
    """

    @staticmethod
    def f1_score(predicted_ids: List[str], ground_truth_ids: List[str]):
        pred = set(predicted_ids)
        gt = set(ground_truth_ids)

        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return float(precision), float(recall), float(f1)

    def evaluate_report(self, report: Dict[str, Any], ground_truth_ids: List[str]) -> Dict[str, Any]:
        """
        RecommendationAgent.create_report() 결과(report)를 받아서
        - 추천된 id 목록 vs 정답 id 목록을 비교해서
        Precision / Recall / F1 계산
        """
        recs = report.get("recommendations", [])
        predicted_ids = [r.get("id") for r in recs if r.get("id")]

        precision, recall, f1 = self.f1_score(predicted_ids, ground_truth_ids)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_predicted": len(predicted_ids),
            "num_ground_truth": len(ground_truth_ids),
            "predicted_ids": predicted_ids,
            "ground_truth_ids": ground_truth_ids,
        }


# ─────────────────────────────────────────────
# 3) 요약 품질 평가용: SummaryEvaluator
# ─────────────────────────────────────────────
class SummaryEvaluator:
    """
    요약 품질 정성/정량 평가 (간단 버전)
    - accuracy: 원문 단어와 겹치는 비율 기반 점수 (대략)
    - redundancy: 문장 길이/반복 정도로 대략 측정
    - clarity: 필요하면 LLM으로 평가 확장 가능 (지금은 placeholder)
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def evaluate_summary(self, original: str, summary: str) -> Dict[str, float]:
        if not original or not summary:
            return {"accuracy": 0.0, "redundancy": 0.0, "clarity": 0.0}

        # 매우 단순한 accuracy 측정: 요약에 등장하는 단어 중 원문에도 있는 비율
        ori_tokens = set(original.split())
        sum_tokens = summary.split()
        if not sum_tokens:
            acc = 0.0
        else:
            overlap = sum(1 for t in sum_tokens if t in ori_tokens)
            acc = overlap / len(sum_tokens)

        # redundancy: 너무 길면 점수 낮게 (대략적인 예시)
        length_penalty = min(len(summary) / 500.0, 1.0)  # 500자 이상이면 패널티 최댓값
        redundancy = max(0.0, 1.0 - length_penalty)

        # clarity: 일단 0.8 고정 (원하면 LLM 기반으로 확장 가능)
        clarity = 0.8

        return {
            "accuracy": float(acc),
            "redundancy": float(redundancy),
            "clarity": float(clarity),
        }


# ─────────────────────────────────────────────
# 4) RAG 전체 품질 평가용: RAGEvaluator
#    - Grounding Precision
#    - BERTScore(F1)
#    - Self-consistency
# ─────────────────────────────────────────────
class RAGEvaluator:
    """
    RAG 시스템 오프라인 평가용 클래스
      - Grounding Precision
      - BERTScore(F1)
      - Self-consistency
    """

    def __init__(self, rag_system, llm_client):
        self.rag = rag_system
        self.llm = llm_client

    # ───────── Grounding Precision ─────────
    def compute_grounding_precision(self, answer: str, retrieved_docs: List[str]) -> float:
        """
        answer 안의 문장들 중에서,
        검색된 문서(retrieved_docs)에 실제로 등장하는 문장이 몇 %인지 계산
        (아주 단순한 버전)
        """
        if not answer or not retrieved_docs:
            return 0.0

        docs_text = " ".join(retrieved_docs).lower()
        sentences = [s.strip() for s in answer.split(".") if s.strip()]

        if not sentences:
            return 0.0

        hit = 0
        for sent in sentences:
            if sent.lower() in docs_text:
                hit += 1

        return hit / len(sentences)

    # ───────── Self-consistency ─────────
    def compute_self_consistency(self, query: str, n: int = 3) -> float:
        """
        같은 질문을 n번 물어보고, 인접 답변들끼리 문자열 유사도를 계산
        """
        answers = []
        for _ in range(n):
            ans = self.llm.generate(prompt=query, system_prompt=None, max_tokens=500)
            answers.append(ans or "")

        if len(answers) < 2:
            return 0.0

        sims = []
        for i in range(len(answers) - 1):
            s = difflib.SequenceMatcher(None, answers[i], answers[i + 1]).ratio()
            sims.append(s)

        return float(mean(sims)) if sims else 0.0

    # ───────── 통합 배치 평가 ─────────
    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        samples 예시:
        [
            {
                "id": "Q1",
                "query": "부산 청년 창업지원 사업 알려줘",
                "reference": "부산 지역 청년을 대상으로 하는 창업지원 사업은 ..."
            },
            ...
        ]
        """
        details = []
        grounding_list = []
        bert_f1_list = []
        sc_list = []

        for sample in samples:
            qid = sample.get("id")
            query = sample.get("query", "")
            reference = sample.get("reference", "")

            # 1) RAG 검색
            retrieved = self.rag.search(query, top_k=5)
            # Document.text에서 내용 추출
            docs_text = []
            for r in retrieved:
                doc = r.get("document")
                text = getattr(doc, "text", "")
                if text:
                    docs_text.append(text)

            # 2) LLM 답변 생성 (RAG 기반이라는 system_prompt)
            system_prompt = (
                "아래 검색된 문서 내용에 기반해서만, 사실 기반으로 간단히 답변하세요. "
                "문서에 없는 내용은 추측해서 만들지 마세요."
            )
            answer = self.llm.generate(
                prompt=query,
                system_prompt=system_prompt,
                max_tokens=600,
            )

            # 3) Grounding Precision
            gp = self.compute_grounding_precision(answer, docs_text)
            grounding_list.append(gp)

            # 4) BERTScore(F1) (설치 안 되어 있으면 0.0)
            if BERTSCORE_AVAILABLE and reference:
                _, _, f1 = bertscore([answer], [reference], lang="ko")
                bert_f1 = float(f1[0])
            else:
                bert_f1 = 0.0
            bert_f1_list.append(bert_f1)

            # 5) Self-consistency
            sc = self.compute_self_consistency(query, n=3)
            sc_list.append(sc)

            details.append({
                "id": qid,
                "query": query,
                "answer": answer,
                "reference": reference,
                "grounding_precision": gp,
                "bertscore_f1": bert_f1,
                "self_consistency": sc,
            })

        return {
            "metrics": {
                "grounding_precision_avg": float(mean(grounding_list)) if grounding_list else 0.0,
                "bertscore_f1_avg": float(mean(bert_f1_list)) if bert_f1_list else 0.0,
                "self_consistency_avg": float(mean(sc_list)) if sc_list else 0.0,
            },
            "details": details,
        }
    
    def evaluate_single(self, query: str, reference: str, sample_id: str = "test") -> Dict[str, Any]:
        """
        한 번 테스트할 때 바로 평가하고 싶을 때 쓰는 헬퍼 함수
        """
        samples = [
            {
                "id": sample_id,
                "query": query,
                "reference": reference,
            }
        ]
        return self.evaluate_batch(samples)
