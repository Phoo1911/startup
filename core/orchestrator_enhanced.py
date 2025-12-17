# core/orchestrator_enhanced.py
"""
Agentic AI 강화 Orchestrator
- ReAct 패턴 적용
- 강화된 도구 시스템
- 논문용 평가 통합
"""

from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

from config.settings import Config
from models.data import UserProfile
from core.llm_client import LLMClient
from core.rag_system import RAGSystem
from core.planner import AgenticPlanner
from core.tools import ToolRegistry
from core.tools_enhanced import (
    create_advanced_search_tool,
    create_multi_stage_filter,
    create_ml_ranking_tool,
    create_analytics_tool
)
from core.evaluation_suite import ComprehensiveEvaluationSuite

from agents.data_collector import DataCollectionAgent
from agents.rag_builder import RAGBuilderAgent


class EnhancedAgenticOrchestrator:
    """강화된 Agentic AI Orchestrator"""

    def __init__(self, service_key: str, llm_api_key: str = None):
        print("\n" + "=" * 80)
        print("🚀 Enhanced Agentic AI 시스템 초기화")
        print("=" * 80)

        # LLM 준비
        try:
            self.llm_client = LLMClient(api_key=llm_api_key)
            print("✅ LLM 준비 완료")
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {e}")
            self.llm_client = None

        # RAG 준비
        try:
            self.rag = RAGSystem(
                embedding_model=Config.EMBEDDING_MODEL,
                provider=Config.EMBEDDING_PROVIDER,
            )
            print(f"✅ RAG 준비 완료")
        except Exception as e:
            print(f"❌ RAG 초기화 실패: {e}")
            raise

        # 데이터 에이전트
        self.data_agent = DataCollectionAgent(service_key, self.llm_client)
        self.rag_builder = RAGBuilderAgent(self.rag, self.llm_client)

        # 도구 시스템 (강화)
        self.tool_registry = ToolRegistry()
        self._setup_enhanced_tools()
        print(f"✅ {len(self.tool_registry.list_tools())}개 도구 등록")

        # Agentic 플래너
        if self.llm_client:
            self.agentic_planner = AgenticPlanner(
                self.llm_client,
                self.tool_registry
            )
            print("✅ Agentic 플래너 준비")
        else:
            self.agentic_planner = None
            print("⚠️ Agentic 플래너 비활성화 (LLM 필요)")

        # 평가 Suite
        if self.llm_client:
            self.eval_suite = ComprehensiveEvaluationSuite(
                self,
                self.llm_client
            )
            print("✅ 평가 Suite 준비")
        else:
            self.eval_suite = None

        # 캐시
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._index_ready = False

        print("=" * 80)
        print("✅ Enhanced Agentic AI 준비 완료!")
        print("=" * 80 + "\n")

    def _setup_enhanced_tools(self):
        """강화된 도구 등록"""
        try:
            # 1️⃣ 고급 검색 (API 파라미터 활용)
            advanced_search = create_advanced_search_tool(
                self.rag,
                self.data_agent
            )
            self.tool_registry.register(advanced_search)

            # 2️⃣ 다단계 필터링
            multi_filter = create_multi_stage_filter()
            self.tool_registry.register(multi_filter)

            # 3️⃣ ML 기반 랭킹
            ml_ranking = create_ml_ranking_tool(self.rag.embed_fn)
            self.tool_registry.register(ml_ranking)

            # 4️⃣ 분석 도구
            analytics = create_analytics_tool()
            self.tool_registry.register(analytics)

        except Exception as e:
            print(f"⚠️ 도구 등록 실패: {e}")

    def ensure_index(self, use_cache: bool = True):
        """RAG 인덱스 준비"""
        if use_cache and self._index_ready:
            print("✅ 인덱스 재사용")
            return

        print("\n" + "=" * 80)
        print("📚 RAG 인덱스 준비")
        print("=" * 80)

        try:
            import pickle

            # RAW DATA
            raw_cache = self.cache_dir / "raw_data.pkl"
            if use_cache and raw_cache.exists():
                print("💾 RAW 데이터 로드")
                with open(raw_cache, "rb") as f:
                    raw_data = pickle.load(f)
            else:
                print("📡 RAW 데이터 수집")
                raw_data = self.data_agent.collect_all(
                    max_pages=Config.MAX_PAGES_PER_ENDPOINT
                )
                with open(raw_cache, "wb") as f:
                    pickle.dump(raw_data, f)

            # RAG INDEX
            faiss_path = self.cache_dir / "rag_index.faiss"
            docs_path = self.cache_dir / "rag_index.docs.pkl"

            if use_cache and faiss_path.exists() and docs_path.exists():
                print("💾 RAG 인덱스 로드")
                self.rag.load(str(self.cache_dir / "rag_index"))
            else:
                print("🔨 RAG 인덱스 구축")
                self.rag_builder.build_index(raw_data)
                self.rag.save(str(self.cache_dir / "rag_index"))

            self._index_ready = True
            print("✅ 인덱스 준비 완료")

        except Exception as e:
            print(f"❌ 인덱스 준비 실패: {e}")
            raise
    

    def run(
        self,
        profile: UserProfile,
        top_n: int = 10,
        use_cache: bool = True,
    ) -> Dict:
        """
        Baseline 추천 (Semantic Matching)
        기존 orchestrator의 run 메서드와 호환
        """
        print("\n" + "=" * 80)
        print("🎯 Baseline 추천 시작 (Semantic Matching)")
        print("=" * 80)

        try:
            self.ensure_index(use_cache)

            from agents.semantic_matcher import SemanticMatchingAgent
            from agents.llm_reasoner import LLMReasoningAgent
            from agents.recommender import RecommendationAgent

            # Semantic Matcher
            semantic_matcher = SemanticMatchingAgent(self.rag, self.llm_client)
            
            # LLM Reasoner (있으면)
            llm_reasoner = None
            if self.llm_client:
                llm_reasoner = LLMReasoningAgent(self.llm_client)
            
            # Recommender
            recommender = RecommendationAgent(self.llm_client)

            # STEP 1: Semantic Matching
            print("\n🔍 STEP 1: Semantic Matching")
            print("-" * 80)

            desired_types = getattr(profile, "desired_data_types", None)

            matches = semantic_matcher.match(
                profile=profile,
                top_k=top_n * 2,
                exclude_closed=True,
                desired_data_types=desired_types,
            )
            print(f"✅ {len(matches)}개 후보 추출")

            # STEP 2: LLM 분석 (선택)
            llm_summary = None
            if llm_reasoner and matches:
                print("\n🧠 STEP 2: LLM 상세 분석")
                print("-" * 80)
                try:
                    matches = llm_reasoner.enhance_matches(matches[:top_n], profile)
                    llm_summary = llm_reasoner.generate_summary(matches, profile)
                    print("✅ LLM 분석 완료")
                except Exception as e:
                    print(f"⚠️ LLM 분석 실패: {e}")

            # STEP 3: 최종 리포트
            print("\n📊 STEP 3: 리포트 생성")
            print("-" * 80)

            report = recommender.create_report(
                matches=matches,
                profile=profile,
                llm_summary=llm_summary,
                top_n=top_n,
            )

            print("\n" + "=" * 80)
            print(f"✅ 완료! {len(report.get('recommendations', []))}개 추천")
            print("=" * 80 + "\n")

            return report

        except Exception as e:
            print(f"\n❌ 추천 실행 실패: {e}")
            import traceback
            traceback.print_exc()

            return {
                "status": "ERROR",
                "message": str(e),
                "recommendations": [],
                "total_matches": 0,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }


    # ═══════════════════════════════════════════════
    # 🤖 Agentic AI 추천
    # ═══════════════════════════════════════════════

    def run_agentic(
        self,
        profile: UserProfile,
        top_n: int = 10,
        use_cache: bool = True
    ) -> Dict:
        """
        Agentic AI 추천 (ReAct 패턴)
        """
        if not self.agentic_planner:
            raise RuntimeError("Agentic 플래너가 없습니다 (LLM 필요)")

        print("\n" + "=" * 80)
        print("🤖 Agentic AI 추천 시작 (ReAct)")
        print("=" * 80)

        try:
            self.ensure_index(use_cache)

            # 사용자 프로필 준비
            user_profile_dict = {
                "name": profile.name,
                "age": profile.age,
                "region": profile.region,
                "business_stage": profile.business_stage,
                "business_field": profile.business_field,
                "target_type": profile.target_type,
                "is_disabled": getattr(profile, "is_disabled", False),
                "desired_data_types": getattr(profile, "desired_data_types", None),
            }

            # Agentic 실행
            result = self.agentic_planner.plan_and_execute(
                user_profile=user_profile_dict,
                task=f"{profile.name}님에게 적합한 창업지원 {top_n}개 추천",
                context={"top_n": top_n}
            )

            recommendations = result.get("recommendations", [])
            
            # 리포트 포맷
            report = {
                "mode": "agentic_react",
                "status": result.get("status"),
                "profile": user_profile_dict,
                "recommendations": recommendations[:top_n],
                "total_matches": len(recommendations),
                "agent_steps": result.get("agent_steps", []),
                "final_state": result.get("final_state", {}),
                "generated_at": datetime.now().isoformat()
            }

            print("\n" + "=" * 80)
            print(f"✅ 완료! {len(recommendations)}개 추천")
            print(f"   총 {result.get('total_steps', 0)}단계 실행")
            print("=" * 80 + "\n")

            return report

        except Exception as e:
            print(f"\n❌ Agentic 추천 실패: {e}")
            import traceback
            traceback.print_exc()

            return {
                "mode": "agentic",
                "status": "ERROR",
                "message": str(e),
                "recommendations": [],
                "total_matches": 0,
                "generated_at": datetime.now().isoformat()
            }

    # ═══════════════════════════════════════════════
    # 📊 평가 실행
    # ═══════════════════════════════════════════════

    def run_evaluation(
        self,
        test_cases: List[Dict],
        output_dir: str = "eval_results"
    ) -> Dict:
        """
        종합 평가 실행
        
        test_cases = [
            {
                "profile": UserProfile(...),
                "ground_truth_ids": ["DOC1", "DOC2", ...],
                "query_id": "Q1"
            },
            ...
        ]
        """
        if not self.eval_suite:
            raise RuntimeError("평가 Suite가 없습니다 (LLM 필요)")

        print("\n" + "=" * 80)
        print("🔬 종합 평가 시작")
        print("=" * 80)

        # 인덱스 준비
        self.ensure_index(use_cache=True)

        # 방법론 비교
        comparison = self.eval_suite.compare_methods(test_cases)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"eval_{timestamp}.json"
        self.eval_suite.save_results(comparison, str(output_path))

        return comparison

    def save_report(self, report: Dict, output_dir: str = "reports"):
        """리포트 저장"""
        import json

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = report.get("mode", "unknown")
        json_path = output_path / f"{mode}_report_{timestamp}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"💾 저장: {json_path}")
        return json_path


# ═══════════════════════════════════════════════
# 🧪 자동 실험 러너
# ═══════════════════════════════════════════════

def run_ablation_study(orchestrator, test_profiles: List[UserProfile]):
    """
    Ablation Study: 각 구성 요소의 기여도 평가
    
    1. Baseline (RAG only)
    2. +Filtering
    3. +Ranking
    4. +Agentic AI (Full)
    """
    from core.ground_truth import build_rule_based_ground_truth
    
    print("\n" + "=" * 80)
    print("🔬 Ablation Study 시작")
    print("=" * 80)

    results = {}

    for i, profile in enumerate(test_profiles, 1):
        print(f"\n[{i}/{len(test_profiles)}] 프로필 평가 중...")

        # Ground Truth 생성
        gt_ids = build_rule_based_ground_truth(
            profile=profile,
            rag_system=orchestrator.rag,
            base_query="창업 지원사업",
            top_k=100,
            desired_data_types=getattr(profile, "desired_data_types", None)
        )

        # 실험 (예: Agentic만)
        report = orchestrator.run_agentic(profile, top_n=10)
        
        predicted_ids = [rec.get("id") for rec in report.get("recommendations", [])]

        # 평가
        from core.evaluation_suite import RecommendationEvaluator
        evaluator = RecommendationEvaluator()
        
        metrics = evaluator.evaluate_single(predicted_ids, gt_ids, k=10)

        results[f"profile_{i}"] = {
            "profile": profile.to_dict(),
            "metrics": metrics,
            "predicted_count": len(predicted_ids),
            "ground_truth_count": len(gt_ids)
        }

    return results


