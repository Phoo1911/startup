# core/orchestrator.py
"""
Hybrid Orchestrator: Semantic Matching + Agentic AI (안전성 개선)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from config.settings import Config
from models.data import UserProfile, MatchResult
from core.llm_client import LLMClient
from core.rag_system import RAGSystem
from core.tools import (
    ToolRegistry,
    create_search_tool,
    create_filter_tool,
    create_analysis_tool,
    create_ranking_tool,
    create_summary_tool,
)
from core.evaluator import MatchingEvaluator
from core.ground_truth import build_rule_based_ground_truth
from core.experiment_logger import ExperimentLogger  # 선택 사용

from agents.data_collector import DataCollectionAgent
from agents.rag_builder import RAGBuilderAgent
from agents.semantic_matcher import SemanticMatchingAgent
from agents.llm_reasoner import LLMReasoningAgent
from agents.recommender import RecommendationAgent
from agents.agentic_base import ChatbotAgenticAgent, RecommendationAgenticAgent


class AgenticOrchestrator:
    """
    Hybrid Orchestrator with Enhanced Safety
    """

    def __init__(self, service_key: str, llm_api_key: str = None):
        print("\n" + "=" * 80)
        print("🤖 Hybrid AI 시스템 초기화 (Semantic + Agentic)")
        print("=" * 80)

        # ───────── LLM 준비 ─────────
        try:
            self.llm_client = LLMClient(api_key=llm_api_key)
            print("✅ LLM 준비")
        except Exception as e:
            print(f"⚠️  LLM 초기화 실패: {e}")
            self.llm_client = None

        # ───────── RAG 준비 ─────────
        try:
            provider = getattr(Config, "EMBEDDING_PROVIDER", "faiss")
            self.rag = RAGSystem(
                embedding_model=Config.EMBEDDING_MODEL,
                provider=provider,
            )
            print(f"✅ RAG 준비 (provider={provider}, model={Config.EMBEDDING_MODEL})")
        except Exception as e:
            print(f"❌ RAG 초기화 실패: {e}")
            raise

        # Semantic Matching + Reasoner
        self.semantic_matcher = SemanticMatchingAgent(self.rag, self.llm_client)
        self.llm_reasoner = (
            LLMReasoningAgent(self.llm_client) if self.llm_client else None
        )
        self.recommender = RecommendationAgent(self.llm_client)
        print("✅ Semantic Matcher 준비")

        # Agentic AI 도구 시스템
        self.tool_registry = ToolRegistry()
        self._setup_tools()
        print(f"✅ {len(self.tool_registry.list_tools())}개 도구 등록")

        # 데이터 수집 + RAG 빌더
        self.data_agent = DataCollectionAgent(service_key, self.llm_client)
        self.rag_builder = RAGBuilderAgent(self.rag, self.llm_client)

        # Agentic 에이전트들
        if self.llm_client:
            try:
                self.chatbot_agent = ChatbotAgenticAgent(
                    self.llm_client,
                    self.tool_registry,
                )
                print("✅ Agentic 챗봇 준비")

                self.agentic_recommender = RecommendationAgenticAgent(
                    self.llm_client,
                    self.tool_registry,
                )
                print("✅ Agentic 추천 에이전트 준비")
            except Exception as e:
                print(f"⚠️  Agentic 에이전트 초기화 실패: {e}")
                self.chatbot_agent = None
                self.agentic_recommender = None
        else:
            self.chatbot_agent = None
            self.agentic_recommender = None

        # 캐시 관련
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.raw_cache_path = self.cache_dir / "raw_data.pkl"
        self.rag_cache_path = self.cache_dir / "rag_index.pkl"

        self._index_ready = False
        self.last_report: Optional[Dict] = None

        self.matching_evaluator = MatchingEvaluator()
        self.experiment_logger = ExperimentLogger()

        print("=" * 80)
        print("✅ Hybrid 시스템 준비 완료!")
        print("=" * 80 + "\n")

    # ───────────────── 도구 등록 ─────────────────
    def _setup_tools(self):
        """Agentic AI 도구 등록"""
        try:
            search_tool = create_search_tool(self.rag)
            self.tool_registry.register(search_tool)

            filter_tool = create_filter_tool()
            self.tool_registry.register(filter_tool)

            ranking_tool = create_ranking_tool()
            self.tool_registry.register(ranking_tool)

            if self.llm_client:
                analysis_tool = create_analysis_tool(self.llm_client)
                self.tool_registry.register(analysis_tool)

                summary_tool = create_summary_tool(self.llm_client)
                self.tool_registry.register(summary_tool)
        except Exception as e:
            print(f"⚠️  도구 등록 실패: {e}")

    # ───────────────── 인덱스 준비 ─────────────────
    def ensure_index(self, use_cache: bool = True):
        """RAG 인덱스 준비 (안전성 강화)"""
        if use_cache and self._index_ready:
            print("✅ 인덱스 재사용 (이미 메모리에 로드됨)")
            return

        print("\n" + "=" * 80)
        print("📚 RAG 인덱스 준비")
        print("=" * 80)

        try:
            # 1) RAW DATA LOAD (캐시 유무)
            if use_cache and self.raw_cache_path.exists():
                print("💾 RAW 데이터 캐시 로드")
                with open(self.raw_cache_path, "rb") as f:
                    raw_data = pickle.load(f)
            else:
                print("📡 RAW 데이터 수집")
                raw_data = self.data_agent.collect_all(
                    max_pages=Config.MAX_PAGES_PER_ENDPOINT
                )
                with open(self.raw_cache_path, "wb") as f:
                    pickle.dump(raw_data, f)

            # 2) RAG INDEX LOAD / BUILD
            faiss_path = Path("cache/rag_index.faiss")
            docs_path = Path("cache/rag_index.docs.pkl")

            cache_exists = faiss_path.exists() and docs_path.exists()

            if use_cache and cache_exists:
                print("💾 RAG 인덱스 로드 (FAISS + 문서)")
                self.rag.load("cache/rag_index")
            else:
                print("🔨 RAG 인덱스 신규 구축")
                self.rag_builder.build_index(raw_data)
                self.rag.save("cache/rag_index")

            self._index_ready = True

            print("=" * 80)
            print("✅ 인덱스 준비 완료")
            print("=" * 80 + "\n")

        except Exception as e:
            print(f"❌ 인덱스 준비 실패: {e}")
            import traceback

            traceback.print_exc()
            raise

    # ───────────────── Hybrid run ─────────────────
    def run(
        self,
        profile: UserProfile,
        top_n: int = 10,
        use_cache: bool = True,
    ) -> Dict:
        """
        Hybrid 추천 실행 (Semantic + LLM Reasoning)
        """
        print("\n" + "=" * 80)
        print("🎯 Hybrid 추천 시작 (Semantic + Agentic)")
        print("=" * 80)

        try:
            self.ensure_index(use_cache)

            # STEP 1: Semantic Matching
            print("\n🔍 STEP 1: Semantic Matching")
            print("-" * 80)

            desired_types = getattr(profile, "desired_data_types", None)

            matches: List[MatchResult] = self.semantic_matcher.match(
                profile=profile,
                top_k=top_n * 2,
                exclude_closed=True,
                desired_data_types=desired_types,
            )
            print(f"✅ {len(matches)}개 후보 추출")

            # STEP 2: LLM 분석 (선택)
            llm_summary: Optional[str] = None

            if self.llm_reasoner and matches:
                print("\n🧠 STEP 2: LLM 상세 분석")
                print("-" * 80)
                try:
                    matches = self.llm_reasoner.enhance_matches(
                        matches[:top_n], profile
                    )
                    llm_summary = self.llm_reasoner.generate_summary(
                        matches, profile
                    )
                    print("✅ LLM 분석 완료")
                except Exception as e:
                    print(f"⚠️  LLM 분석 실패: {e}")

            # STEP 3: 최종 리포트 생성
            print("\n📊 STEP 3: 리포트 생성")
            print("-" * 80)

            report = self.recommender.create_report(
                matches=matches,
                profile=profile,
                llm_summary=llm_summary,
                top_n=top_n,
            )

            self.last_report = report

            print("\n" + "=" * 80)
            print(f"✅ 완료! {len(report.get('recommendations', []))}개 추천")
            print("=" * 80 + "\n")

            # 필요하면 여기서도 실험 로그 저장
            if Config.EVAL_LOG_ENABLED:
                self.experiment_logger.log_run(
                    mode="hybrid",
                    profile=profile.to_dict(),
                    report=report,
                    metrics=None,
                )

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

    # ───────────────── Agentic run ─────────────────
    def run_agentic(
        self,
        profile: UserProfile,
        top_n: int = 10,
        use_cache: bool = True,
    ) -> Dict:
        """
        Agentic 추천 실행 (검색 + 필터 + LLM Rerank)
        """
        if not self.agentic_recommender:
            raise RuntimeError(
                "LLM이 없어 Agentic 추천을 사용할 수 없습니다."
            )

        print("\n" + "=" * 80)
        print("🎯 Agentic 추천 시작 (검색 + 필터 + LLM Rerank)")
        print("=" * 80)

        try:
            self.ensure_index(use_cache)

            desired_types = getattr(profile, "desired_data_types", None)

            user_profile_dict = {
                "name": profile.name if profile else "",
                "age": profile.age if profile else 0,
                "region": profile.region if profile else "전국",
                "business_stage": profile.business_stage if profile else "",
                "business_field": profile.business_field if profile else "",
                "target_type": profile.target_type if profile else "",
                "is_disabled": getattr(profile, "is_disabled", False),
                "desired_data_types": desired_types,
            }

            result = self.agentic_recommender.recommend(
                user_profile=user_profile_dict,
                top_n=top_n,
            )

            recommendations = result.get("recommendations", [])
            iterations = result.get("iterations", 0)
            thoughts = result.get("agent_thoughts", [])

            report = {
                "mode": "agentic_search_filter_rank",
                "status": "SUCCESS",
                "profile": user_profile_dict,
                "recommendations": recommendations,
                "total_matches": len(recommendations),
                "iterations": iterations,
                "agent_thoughts": thoughts,
                "generated_at": datetime.now().isoformat(),
            }

            self.last_report = report

            print("\n" + "=" * 80)
            print(
                f"✅ Agentic 완료! {len(recommendations)}개 추천 (iterations={iterations})"
            )
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
                "generated_at": datetime.now().isoformat(),
            }

    # ───────────────── 챗봇 ─────────────────
    def chat(
        self,
        profile: UserProfile,
        question: str,
        history: Optional[list] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Agentic AI 챗봇 (안전성 강화)"""
        try:
            self.ensure_index(use_cache=True)

            if not self.chatbot_agent:
                from agents.chatbot import ChatbotAgent

                fallback_bot = ChatbotAgent(self.rag, None)
                return fallback_bot.chat(
                    profile=profile,
                    question=question,
                    history=history,
                    category=category,
                    **kwargs,
                )

            user_profile_dict = {
                "name": profile.name if profile else "",
                "age": profile.age if profile else 0,
                "region": profile.region if profile else "전국",
                "business_stage": profile.business_stage if profile else "",
                "business_field": profile.business_field if profile else "",
                "target_type": profile.target_type if profile else "",
                "is_disabled": getattr(profile, "is_disabled", None),
                "desired_data_types": getattr(
                    profile, "desired_data_types", None
                ),
            }

            answer = self.chatbot_agent.chat(
                user_profile=user_profile_dict,
                question=question,
                category=category,
            )

            return answer

        except Exception as e:
            print(f"❌ 챗봇 오류: {e}")
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}"

    # ───────────────── run_with_eval ─────────────────
    def run_with_eval(
        self,
        profile: UserProfile,
        top_n: int = 10,
        use_cache: bool = True,
        base_query: str = "창업 지원사업",
    ) -> Dict:
        """
        Hybrid 추천 + 룰 기반 평가를 한 번에 수행하는 함수
        """
        report = self.run(
            profile=profile,
            top_n=top_n,
            use_cache=use_cache,
        )

        if report.get("status") == "ERROR":
            return report

        try:
            desired_types = getattr(profile, "desired_data_types", None)

            gt_ids = build_rule_based_ground_truth(
                profile=profile,
                rag_system=self.rag,
                base_query=base_query,
                top_k=300,
                desired_data_types=desired_types,
            )

            eval_result = self.matching_evaluator.evaluate_report(
                report=report,
                ground_truth_ids=gt_ids,
            )

            report["matching_eval"] = eval_result

            print("\n📏 추천 적합도 평가 결과")
            print("-" * 80)
            print(f"Precision: {eval_result['precision']:.3f}")
            print(f"Recall   : {eval_result['recall']:.3f}")
            print(f"F1-score : {eval_result['f1']:.3f}")
            print(
                f"추천 {eval_result['num_predicted']}개 / 정답 {eval_result['num_ground_truth']}개"
            )

            # 평가 결과도 로그에 저장하고 싶으면
            if Config.EVAL_LOG_ENABLED:
                self.experiment_logger.log_run(
                    mode="hybrid_eval",
                    profile=profile.to_dict(),
                    report=report,
                    metrics=eval_result,
                )

        except Exception as e:
            print(f"⚠️  매칭 평가 수행 중 오류: {e}")

        return report

    # ───────────────── save_report ─────────────────
    def save_report(self, report: Dict, output_dir: str = "reports"):
        """리포트 JSON 저장"""
        import numpy as np

        def to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"hybrid_report_{timestamp}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=2,
                default=to_serializable,
            )

        print(f"💾 저장: {json_path}")
        return json_path
