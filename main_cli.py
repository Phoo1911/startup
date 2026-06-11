#!/usr/bin/env python3
"""
Agentic AI 시스템 CLI 테스트 (개선 버전)
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from config.settings import Config
from models.data import UserProfile
from legacy_core.orchestrator import AgenticOrchestrator


def test_recommendation():
    """추천 시스템 테스트"""
    print("\n" + "="*80)
    print("🧪 Agentic AI 추천 시스템 테스트")
    print("="*80)
    
    # 설정 검증
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        print("\n💡 .env 파일을 확인하세요:")
        print("   - KISED_SERVICE_KEY")
        print("   - OPENAI_API_KEY")
        return
    
    # 오케스트레이터 초기화
    try:
        print("\n🔧 시스템 초기화 중...")
        orchestrator = AgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY
        )
        print("✅ 오케스트레이터 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 테스트 프로필
    profile = UserProfile(
        name="테스트사용자",
        age=29,
        region="서울",
        business_stage="예비창업자",
        business_field="AI",
        target_type="청년",
        is_veteran=False,
        is_disabled=False,
        additional_context="AI 기반 스타트업 관심",
        desired_data_types=["announcement", "business", "lecture"]
    )
    
    print(f"\n📋 테스트 프로필:")
    print(f"   - 지역: {profile.region}")
    print(f"   - 나이: {profile.age}세")
    print(f"   - 분야: {profile.business_field}")
    print(f"   - 단계: {profile.business_stage}")
    
    # 추천 실행
    try:
        print("\n🚀 추천 시작...")
        report = orchestrator.run(
            profile=profile,
            top_n=5,
            use_cache=True
        )
        print("✅ 추천 완료!")
        
        # 결과 출력
        print("\n" + "="*80)
        print("📊 추천 결과")
        print("="*80)
        
        if report.get("status") == "SUCCESS":
            recommendations = report.get('recommendations', [])
            print(f"\n✅ 총 {len(recommendations)}개 추천")
            
            # 추천 항목 출력
            print("\n📌 상위 추천:")
            for rec in recommendations[:5]:
                print(f"\n{rec.get('rank', '?')}. {rec.get('title', '제목없음')}")
                print(f"   타입: {rec.get('data_type', 'N/A')}")
                print(f"   점수: {rec.get('match_score', 0)}")
                print(f"   지역: {rec.get('region', 'N/A')}")
                print(f"   마감: {rec.get('deadline', 'N/A')}")
                
                # 추천 이유
                reasons = rec.get('reasons', [])
                if reasons:
                    print(f"   이유:")
                    for reason in reasons[:3]:
                        print(f"      - {reason}")
            
            # 리포트 저장
            try:
                report_path = orchestrator.save_report(report)
                print(f"\n💾 리포트 저장됨: {report_path}")
            except Exception as e:
                print(f"\n⚠️  리포트 저장 실패: {e}")
            
        else:
            print("❌ 추천 결과 없음")
            print(f"상태: {report.get('status')}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
        return
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def test_chatbot():
    """챗봇 테스트"""
    print("\n" + "="*80)
    print("🧪 Agentic AI 챗봇 테스트")
    print("="*80)
    
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        return
    
    try:
        orchestrator = AgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY
        )
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    profile = UserProfile(
        name="테스트사용자",
        age=29,
        region="서울",
        business_stage="예비창업자",
        business_field="AI",
        target_type="청년",
        additional_context=""
    )
    
    # 테스트 질문들
    questions = [
        "서울 청년 지원 프로그램 알려줘",
        "AI 분야 창업 교육 뭐 있어?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"💬 질문 {i}: {question}")
        print(f"{'='*80}")
        
        try:
            answer = orchestrator.chat(
                profile=profile,
                question=question,
                category="전체"
            )
            print(f"\n🤖 답변:\n{answer}")
        except Exception as e:
            print(f"❌ 오류: {e}")
            import traceback
            traceback.print_exc()


def interactive_mode():
    """대화형 모드"""
    print("\n" + "="*80)
    print("💬 Agentic AI 대화형 모드")
    print("="*80)
    print("'quit' 또는 'exit'를 입력하면 종료됩니다.\n")
    
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        return
    
    try:
        orchestrator = AgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY
        )
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    # 간단한 프로필
    profile = UserProfile(
        name="사용자",
        age=29,
        region="서울",
        business_stage="예비창업자",
        business_field="AI",
        target_type="청년",
        additional_context=""
    )
    
    while True:
        try:
            question = input("\n👤 질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("👋 종료합니다.")
                break
            
            if not question:
                continue
            
            print("\n🤖 답변 생성 중...")
            answer = orchestrator.chat(
                profile=profile,
                question=question,
                category="전체"
            )
            print(f"\n🤖 {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


def simple_test():
    """간단한 동작 테스트"""
    print("\n" + "="*80)
    print("🔍 간단한 시스템 테스트")
    print("="*80)
    
    try:
        print("\n1️⃣ 설정 확인...")
        Config.validate()
        print("   ✅ 설정 OK")
        
        print("\n2️⃣ 오케스트레이터 초기화...")
        orchestrator = AgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY
        )
        print("   ✅ 초기화 OK")
        
        print("\n3️⃣ RAG 시스템 확인...")
        if orchestrator._index_ready:
            print("   ✅ RAG 인덱스 준비됨")
        else:
            print("   ⚠️  RAG 인덱스 준비 안됨 (자동 생성 예정)")
        
        print("\n4️⃣ LLM 연결 확인...")
        if orchestrator.llm_client:
            print("   ✅ LLM 클라이언트 연결됨")
        else:
            print("   ⚠️  LLM 클라이언트 없음 (기본 기능만 사용)")
        
        print("\n" + "="*80)
        print("✅ 시스템 정상 작동!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic AI 시스템 테스트")
    parser.add_argument(
        "mode",
        choices=["recommend", "chat", "interactive", "test", "all"],
        nargs="?",
        default="test",
        help="테스트 모드 (기본: test)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "recommend":
            test_recommendation()
        elif args.mode == "chat":
            test_chatbot()
        elif args.mode == "interactive":
            interactive_mode()
        elif args.mode == "test":
            simple_test()
        else:  # all
            simple_test()
            print("\n" + "="*80 + "\n")
            test_recommendation()
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()