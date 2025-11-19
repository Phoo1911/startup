"""
오케스트레이터 - 전체 시스템 조율
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

from config.settings import Config
from models.data import UserProfile
from core.llm_client import LLMClient
from core.rag_system import RAGSystem
from agents.data_collector import DataCollectionAgent
from agents.rag_builder import RAGBuilderAgent
from agents.semantic_matcher import SemanticMatchingAgent
from agents.llm_reasoner import LLMReasoningAgent
from agents.recommender import RecommendationAgent
from agents.chatbot import ChatbotAgent

class AgenticOrchestrator:
    """전체 시스템 오케스트레이터"""
    
    def __init__(self, service_key: str, llm_api_key: str = None):
        # LLM 클라이언트
        self.llm_client = None
        if llm_api_key:
            self.llm_client = LLMClient(api_key=llm_api_key)
        
        # RAG 시스템
        self.rag = RAGSystem(embedding_model=Config.EMBEDDING_MODEL)
        
        # 에이전트 초기화
        self.data_agent = DataCollectionAgent(service_key, self.llm_client)
        self.rag_builder = RAGBuilderAgent(self.rag, self.llm_client)
        self.semantic_matcher = SemanticMatchingAgent(self.rag, self.llm_client)
        self.llm_reasoner = LLMReasoningAgent(self.llm_client) if self.llm_client else None
        self.recommender = RecommendationAgent(self.llm_client)
        self.chatbot = ChatbotAgent(self.rag, self.llm_client)
        
        # 최근 리포트 보관
        self.last_report: Dict | None = None
        
        if Config.AGENT_VERBOSE:
            print("="*80)
            print("🤖 Agentic AI + RAG + LLM 시스템 초기화 완료")
            print("="*80)
    
    def run(self, profile: UserProfile, top_n: int = 10, use_cache: bool = True) -> Dict:
        """전체 매칭 프로세스 실행"""
        if Config.AGENT_VERBOSE:
            print(f"\n{'='*80}")
            print(f"🎯 매칭 시작: {profile.name}")
            print(f"{'='*80}\n")
            print("📡 STEP 1: 데이터 수집")
            print("-" * 80)
        
        # STEP 1: 데이터 수집
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        if use_cache and (cache_dir / "raw_data.pkl").exists():
            if Config.AGENT_VERBOSE:
                print("💾 캐시 데이터 사용")
            with open(cache_dir / "raw_data.pkl", 'rb') as f:
                raw_data = pickle.load(f)
        else:
            raw_data = self.data_agent.collect_all(max_pages=3)
            with open(cache_dir / "raw_data.pkl", 'wb') as f:
                pickle.dump(raw_data, f)
        
        # STEP 2: RAG 구축
        if Config.AGENT_VERBOSE:
            print(f"\n📚 STEP 2: RAG 인덱스 구축")
            print("-" * 80)
        
        if use_cache and (cache_dir / "rag_index.pkl").exists():
            if Config.AGENT_VERBOSE:
                print("💾 RAG 인덱스 로드")
            self.rag.load(str(cache_dir / "rag_index.pkl"))
        else:
            self.rag_builder.build_index(raw_data)
            self.rag.save(str(cache_dir / "rag_index.pkl"))
        
        # STEP 3: 의미 매칭
        if Config.AGENT_VERBOSE:
            print(f"\n🎯 STEP 3: 의미 기반 매칭 (RAG)")
            print("-" * 80)
        
        matches = self.semantic_matcher.match(profile, top_k=top_n * 2)
        
        # STEP 4: LLM 분석
        if self.llm_reasoner:
            if Config.AGENT_VERBOSE:
                print(f"\n🧠 STEP 4: LLM 추론")
                print("-" * 80)
            matches = self.llm_reasoner.enhance_matches(matches, profile)
            llm_summary = self.llm_reasoner.generate_summary(matches, profile)
        else:
            llm_summary = None
            if Config.AGENT_VERBOSE:
                print(f"\n⚠️ STEP 4: LLM 비활성화")
        
        # STEP 5: 최종 추천
        if Config.AGENT_VERBOSE:
            print(f"\n✨ STEP 5: 추천 생성")
            print("-" * 80)
        
        report = self.recommender.create_report(matches, profile, llm_summary, top_n)
        self.last_report = report
        
        if Config.AGENT_VERBOSE:
            print(f"\n{'='*80}")
            print("✅ 완료!")
            print(f"{'='*80}\n")
        
        return report
    
    def save_report(self, report: Dict, output_dir: str = "reports"):
        """리포트 저장"""
        def to_serializable(obj):
            """numpy 타입 등을 일반 타입으로 변환"""
            import numpy as np
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
        json_path = output_path / f"report_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=2,
                default=to_serializable
            )
        
        print(f"💾 리포트 저장: {json_path}")
        return json_path

def run_with_auto_refresh(orchestrator, profile, top_n=10, refresh_days=1):
    """자동 새로고침 실행 (하루에 한 번만 새로 수집)"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    meta_path = cache_dir / "meta.json"
    
    use_cache = False
    
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        last_updated = datetime.fromisoformat(meta.get("last_updated"))
        if datetime.now() - last_updated < timedelta(days=refresh_days):
            use_cache = True
    
    # 실행
    report = run_with_auto_refresh(
    orchestrator, 
    profile, 
    top_n=10,
    refresh_days=1  # 1일마다 갱신
)
    
    # 마지막 수집 시간 갱신
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"last_updated": datetime.now().isoformat()}, f)
    
    print(f"🔄 use_cache={use_cache} 로 실행됨")
    return report
