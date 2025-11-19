"""
CLI 실행 진입점
Usage: python main.py --name "김창업" --region "서울" --field "AI"
"""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from config.settings import Config
from models.data import UserProfile
from core.orchestrator import AgenticOrchestrator

def main():
    parser = argparse.ArgumentParser(
        description="🚀 AI 창업지원 매칭 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --name "김창업" --age 29 --region "서울" --field "AI"
  python main.py --name "이스타트" --age 35 --region "부산" --field "제조" --veteran
        """
    )
    
    # 필수 인자
    parser.add_argument('--name', required=True, help='이름')
    parser.add_argument('--age', type=int, default=29, help='나이 (기본: 29)')
    parser.add_argument('--region', required=True, help='지역 (예: 서울, 부산)')
    parser.add_argument('--stage', default='예비창업자', 
                        choices=['예비창업자', '3년이하', '7년이하', '10년이하'],
                        help='창업단계')
    parser.add_argument('--field', required=True, help='사업분야 (예: AI, 제조)')
    parser.add_argument('--target', default='청년',
                        choices=['청년', '여성', '일반', '중장년', '예비창업자'],
                        help='대상유형')
    
    # 선택 인자
    parser.add_argument('--veteran', action='store_true', help='참전유공자')
    parser.add_argument('--disabled', action='store_true', help='장애인')
    parser.add_argument('--context', default='', help='추가 설명')
    
    # 시스템 설정
    parser.add_argument('--top-n', type=int, default=10, help='추천 개수 (기본: 10)')
    parser.add_argument('--no-cache', action='store_true', help='캐시 사용 안 함')
    parser.add_argument('--output', help='결과 JSON 저장 경로')
    parser.add_argument('--verbose', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 설정
    if args.verbose:
        Config.AGENT_VERBOSE = True
    
    # 프로필 생성
    profile = UserProfile(
        name=args.name,
        age=args.age,
        region=args.region,
        business_stage=args.stage,
        business_field=args.field,
        target_type=args.target,
        is_veteran=args.veteran,
        is_disabled=args.disabled,
        additional_context=args.context
    )
    
    print("="*80)
    print(f"🚀 AI 창업지원 매칭 시스템")
    print("="*80)
    print(f"👤 {profile.name} ({profile.age}세)")
    print(f"📍 {profile.region} | 🏢 {profile.business_stage} | 💼 {profile.business_field}")
    print("="*80)
    
    # Orchestrator 실행
    try:
        Config.validate()
        orchestrator = AgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY
        )
        
        report = orchestrator.run(
            profile,
            top_n=args.top_n,
            use_cache=not args.no_cache
        )
        
        # 결과 출력
        print(f"\n✅ 매칭 완료: {report['total_matches']}개 발견")
        print("\n🏆 상위 추천:")
        print("-" * 80)
        
        for rec in report['recommendations'][:5]:
            print(f"\n{rec['rank']}. {rec['title']}")
            print(f"   📊 {rec['match_score']:.1f}점 | 🔥 {rec['priority']} | 📂 {rec['data_type']}")
            print(f"   📅 {rec['deadline']}")
            print(f"   ✓ {rec['reasons'][0] if rec['reasons'] else ''}")
        
        # LLM 요약
        if report.get('llm_summary'):
            print("\n" + "="*80)
            print("🧠 AI 분석:")
            print("-" * 80)
            print(report['llm_summary'])
        
        # 저장
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 결과 저장: {output_path}")
        else:
            # 자동 저장
            json_path = orchestrator.save_report(report)
            print(f"\n💾 결과 저장: {json_path}")
        
        print("\n" + "="*80)
        print("✅ 완료!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
