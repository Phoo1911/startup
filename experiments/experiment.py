#!/usr/bin/env python3
"""
논문용 실험 실행 스크립트
- Baseline vs Agentic 비교
- Ablation Study
- 성능 분석
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from config.settings import Config
from models.data import UserProfile
from core.orchestrator_enhanced import EnhancedAgenticOrchestrator, run_ablation_study
from core.ground_truth import build_rule_based_ground_truth


# ═══════════════════════════════════════════════
# 🧪 테스트 케이스 정의
# ═══════════════════════════════════════════════

def create_test_profiles():
    """다양한 테스트 프로필 생성"""
    
    profiles = [
        # 1. 청년 예비창업자 (서울, AI)
        UserProfile(
            name="테스트_청년_AI",
            age=29,
            region="서울",
            business_stage="예비창업자",
            business_field="AI",
            target_type="청년",
            is_disabled=False,
            desired_data_types=["announcement", "business", "lecture"]
        ),
        
        # 2. 초기창업자 (부산, 제조)
        UserProfile(
            name="테스트_초기_제조",
            age=35,
            region="부산",
            business_stage="3년이하",
            business_field="제조",
            target_type="일반",
            is_disabled=False,
            desired_data_types=["announcement", "business", "space"]
        ),
        
        # 3. 예비창업자 (경기, 유통)
        UserProfile(
            name="테스트_예비_유통",
            age=42,
            region="경기",
            business_stage="예비창업자",
            business_field="유통",
            target_type="중장년",
            is_disabled=False,
            desired_data_types=["announcement", "business"]
        ),
        
        # 4. 장애인 창업자 (대전, ICT)
        UserProfile(
            name="테스트_장애인_ICT",
            age=31,
            region="대전",
            business_stage="예비창업자",
            business_field="ICT",
            target_type="청년",
            is_disabled=True,
            desired_data_types=["announcement", "lecture"]
        ),
        
        # 5. 초기창업자 (광주, 콘텐츠)
        UserProfile(
            name="테스트_초기_콘텐츠",
            age=27,
            region="광주",
            business_stage="3년이하",
            business_field="콘텐츠",
            target_type="청년",
            is_disabled=False,
            desired_data_types=["announcement", "business", "lecture", "space"]
        )
    ]
    
    return profiles


# ═══════════════════════════════════════════════
# 🔬 실험 1: Baseline vs Agentic 비교
# ═══════════════════════════════════════════════

def experiment_baseline_vs_agentic(orchestrator):
    """Baseline (Semantic) vs Agentic AI 비교"""
    
    print("\n" + "="*80)
    print("🔬 실험 1: Baseline vs Agentic AI 비교")
    print("="*80)
    
    profiles = create_test_profiles()
    
    # Ground Truth 생성
    test_cases = []
    for profile in profiles:
        gt_ids = build_rule_based_ground_truth(
            profile=profile,
            rag_system=orchestrator.rag,
            base_query="창업 지원사업",
            top_k=100,
            desired_data_types=profile.desired_data_types
        )
        
        test_cases.append({
            "profile": profile,
            "ground_truth_ids": gt_ids,
            "query_id": profile.name
        })
    
    # 평가 실행
    results = orchestrator.run_evaluation(
        test_cases,
        output_dir="eval_results/exp1_baseline_vs_agentic"
    )
    
    return results


# ═══════════════════════════════════════════════
# 🔬 실험 2: Ablation Study
# ═══════════════════════════════════════════════

def experiment_ablation_study(orchestrator):
    """각 구성 요소의 기여도 분석"""
    
    print("\n" + "="*80)
    print("🔬 실험 2: Ablation Study")
    print("="*80)
    
    profiles = create_test_profiles()[:3]  # 3개만
    
    results = run_ablation_study(orchestrator, profiles)
    
    # 저장
    import json
    from datetime import datetime
    
    output_path = Path("eval_results/exp2_ablation")
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"ablation_{timestamp}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 결과 저장: {json_path}")
    
    return results


# ═══════════════════════════════════════════════
# 🔬 실험 3: 도구 사용 분석
# ═══════════════════════════════════════════════

def experiment_tool_usage_analysis(orchestrator):
    """Agentic AI의 도구 사용 패턴 분석"""
    
    print("\n" + "="*80)
    print("🔬 실험 3: 도구 사용 패턴 분석")
    print("="*80)
    
    profiles = create_test_profiles()
    
    tool_usage_stats = {
        "total_runs": 0,
        "avg_steps": 0,
        "tool_calls": {},
        "success_rate": 0.0
    }
    
    total_steps = 0
    successful_runs = 0
    
    for profile in profiles:
        print(f"\n처리 중: {profile.name}")
        
        try:
            report = orchestrator.run_agentic(profile, top_n=10)
            
            if report.get("status") == "SUCCESS":
                successful_runs += 1
            
            steps = report.get("agent_steps", [])
            total_steps += len(steps)
            
            # 도구 호출 통계
            for step in steps:
                action = step.get("action")
                if action:
                    tool_usage_stats["tool_calls"][action] = \
                        tool_usage_stats["tool_calls"].get(action, 0) + 1
        
        except Exception as e:
            print(f"⚠️ 오류: {e}")
    
    tool_usage_stats["total_runs"] = len(profiles)
    tool_usage_stats["avg_steps"] = total_steps / len(profiles) if profiles else 0
    tool_usage_stats["success_rate"] = successful_runs / len(profiles) if profiles else 0
    
    # 저장
    import json
    from datetime import datetime
    
    output_path = Path("eval_results/exp3_tool_usage")
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"tool_usage_{timestamp}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tool_usage_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과 저장: {json_path}")
    
    print("\n📊 도구 사용 통계:")
    print(f"  - 총 실행: {tool_usage_stats['total_runs']}회")
    print(f"  - 평균 단계: {tool_usage_stats['avg_steps']:.1f}단계")
    print(f"  - 성공률: {tool_usage_stats['success_rate']*100:.1f}%")
    print("\n  도구별 호출 횟수:")
    for tool, count in sorted(
        tool_usage_stats['tool_calls'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"    - {tool}: {count}회")
    
    return tool_usage_stats


# ═══════════════════════════════════════════════
# 🔬 실험 4: 성능 벤치마크
# ═══════════════════════════════════════════════

def experiment_performance_benchmark(orchestrator):
    """레이턴시 및 처리량 측정"""
    
    print("\n" + "="*80)
    print("🔬 실험 4: 성능 벤치마크")
    print("="*80)
    
    import time
    
    profile = UserProfile(
        name="벤치마크",
        age=30,
        region="서울",
        business_stage="예비창업자",
        business_field="AI",
        target_type="청년",
        desired_data_types=["announcement", "business"]
    )
    
    # 워밍업
    print("\n워밍업 중...")
    orchestrator.run_agentic(profile, top_n=5)
    
    # 벤치마크
    print("\n벤치마크 실행 중...")
    n_runs = 10
    latencies = []
    
    for i in range(n_runs):
        start = time.time()
        orchestrator.run_agentic(profile, top_n=10)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        print(f"  Run {i+1}/{n_runs}: {latency:.0f}ms")
    
    # 통계
    import numpy as np
    
    stats = {
        "n_runs": n_runs,
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99))
    }
    
    print("\n📊 레이턴시 통계:")
    print(f"  - 평균: {stats['mean_ms']:.0f}ms")
    print(f"  - 표준편차: {stats['std_ms']:.0f}ms")
    print(f"  - P50: {stats['p50_ms']:.0f}ms")
    print(f"  - P95: {stats['p95_ms']:.0f}ms")
    print(f"  - P99: {stats['p99_ms']:.0f}ms")
    
    # 저장
    import json
    from datetime import datetime
    
    output_path = Path("eval_results/exp4_performance")
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"performance_{timestamp}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과 저장: {json_path}")
    
    return stats


# ═══════════════════════════════════════════════
# 📊 결과 시각화
# ═══════════════════════════════════════════════

def visualize_results(results_dir="eval_results"):
    """결과 시각화 (matplotlib)"""
    
    try:
        import matplotlib.pyplot as plt
        import json
        
        print("\n" + "="*80)
        print("📊 결과 시각화")
        print("="*80)
        
        # 실험 1 결과 로드
        exp1_files = list(Path(results_dir).glob("exp1_*/eval_*.json"))
        if not exp1_files:
            print("⚠️ 실험 1 결과 없음")
            return
        
        with open(exp1_files[-1], 'r') as f:
            comparison = json.load(f)
        
        # 그래프 생성
        metrics = ['precision', 'recall', 'f1', 'ndcg']
        baseline_vals = [
            comparison['baseline']['recommendation'][m] 
            for m in metrics
        ]
        agentic_vals = [
            comparison['agentic']['recommendation'][m]
            for m in metrics
        ]
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline')
        ax.bar([i + width/2 for i in x], agentic_vals, width, label='Agentic AI')
        
        ax.set_ylabel('Score')
        ax.set_title('Baseline vs Agentic AI Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 저장
        output_path = Path(results_dir) / "comparison_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 차트 저장: {output_path}")
        
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib 없음 (pip install matplotlib)")
    except Exception as e:
        print(f"⚠️ 시각화 실패: {e}")


# ═══════════════════════════════════════════════
# 🚀 메인 실행
# ═══════════════════════════════════════════════

def main():
    """전체 실험 실행"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="논문용 실험 실행")
    parser.add_argument(
        "experiment",
        choices=["all", "exp1", "exp2", "exp3", "exp4", "viz"],
        default="all",
        help="실행할 실험"
    )
    
    args = parser.parse_args()
    
    try:
        # 설정 검증
        Config.validate()
        
        # Orchestrator 초기화
        print("\n🔧 시스템 초기화...")
        orchestrator = EnhancedAgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY
        )
        
        # 실험 실행
        if args.experiment == "all":
            experiment_baseline_vs_agentic(orchestrator)
            experiment_ablation_study(orchestrator)
            experiment_tool_usage_analysis(orchestrator)
            experiment_performance_benchmark(orchestrator)
            visualize_results()
        
        elif args.experiment == "exp1":
            experiment_baseline_vs_agentic(orchestrator)
        
        elif args.experiment == "exp2":
            experiment_ablation_study(orchestrator)
        
        elif args.experiment == "exp3":
            experiment_tool_usage_analysis(orchestrator)
        
        elif args.experiment == "exp4":
            experiment_performance_benchmark(orchestrator)
        
        elif args.experiment == "viz":
            visualize_results()
        
        print("\n" + "="*80)
        print("✅ 모든 실험 완료!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자가 중단했습니다.")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()