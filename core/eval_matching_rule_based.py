# scripts/eval_matching_rule_based.py

from core.rag_system import RAGSystem
from core.llm_client import LLMClient
from core.evaluator import MatchingEvaluator
from core.ground_truth import build_rule_based_ground_truth
from models.data import UserProfile
from agents.semantic_matcher import SemanticMatchingAgent
from agents.recommender import RecommendationAgent
from config.settings import Config


def main():
    # 1) 설정 및 객체 준비
    print("🔧 시스템 초기화 중...")

    rag = RAGSystem(
        embedding_model=Config.EMBEDDING_MODEL,
        provider=Config.EMBEDDING_PROVIDER,
    )

    # ✅ orchestrator.ensure_index()에서 저장한 인덱스 로드
    rag.load("cache/rag_index")

    # LLM은 있어도 되고, 없어도 됨
    llm = LLMClient()  # Config 기반으로 OpenAI / Local 등 선택

    semantic_matcher = SemanticMatchingAgent(rag, llm)
    recommender = RecommendationAgent(llm)
    matching_eval = MatchingEvaluator()

    # 2) 테스트용 프로필 (예시)
    profile = UserProfile(
        name="테스트 유저",
        age=29,
        region="부산",
        business_stage="예비창업",
        business_field="ICT",
        target_type="청년",
    )

    desired_types = ["announcement"]  # 문서의 data_type 값과 일치해야 함

    # 3) 추천 실행
    print("🔍 SemanticMatching 실행...")
    matches = semantic_matcher.match(
        profile=profile,
        top_k=30,
        exclude_closed=True,
        desired_data_types=desired_types,
    )
    print(f"  → 후보 {len(matches)}개")

    print("📊 RecommendationAgent.create_report 실행...")
    report = recommender.create_report(
        matches=matches,
        profile=profile,
        llm_summary=None,
        top_n=10,
    )
    print(f"  → 추천 결과 {len(report.get('recommendations', []))}개")

    # 4) 룰 기반 정답 생성
    print("🧠 규칙 기반 ground truth 생성...")
    ground_truth_ids = build_rule_based_ground_truth(
        profile=profile,
        rag_system=rag,
        base_query="창업 지원사업",
        top_k=300,
        desired_data_types=desired_types,
    )
    print(f"  → ground truth 크기: {len(ground_truth_ids)}개")

    # 5) 추천 적합도 평가
    print("✅ MatchingEvaluator.evaluate_report 실행...")
    eval_result = matching_eval.evaluate_report(report, ground_truth_ids)

    print("\n===== 추천 적합도 평가 결과 =====")
    print(f"Precision: {eval_result['precision']:.3f}")
    print(f"Recall   : {eval_result['recall']:.3f}")
    print(f"F1-score : {eval_result['f1']:.3f}")
    print(
        f"추천 개수: {eval_result['num_predicted']}, 정답 개수: {eval_result['num_ground_truth']}"
    )

    # 필요하면 report에 평가 정보 붙여서 JSON으로 저장
    report["matching_eval"] = eval_result
    # 예: orchestrator.save_report(report) 등과 연동 가능


if __name__ == "__main__":
    main()
