"""
Streamlit 기반 웹 UI
실행: streamlit run web/streamlit_app.py

"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from models.data import UserProfile
from core.orchestrator import AgenticOrchestrator
import pandas as pd
import json
import numpy as np  

def to_serializable(obj):
    """JSON으로 저장 가능하도록 numpy 타입을 기본 파이썬 타입으로 변환"""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

# 페이지 설정
st.set_page_config(
    page_title="🚀 창업지원 매칭 시스템",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .match-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .priority-high { border-left-color: #ef5350; }
    .priority-medium { border-left-color: #ffa726; }
    .priority-low { border-left-color: #66bb6a; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def init_orchestrator():
    """Orchestrator 초기화"""
    if st.session_state.orchestrator is None:
        try:
            Config.validate()
            st.session_state.orchestrator = AgenticOrchestrator(
                service_key=Config.SERVICE_KEY,
                llm_api_key=Config.LLM_API_KEY
            )
            st.success("✅ 시스템 초기화 완료!")
        except Exception as e:
            st.error(f"❌ 초기화 실패: {e}")
            return False
    return True

def display_match_card(match, rank):
    """매칭 결과 카드 표시"""
    priority_class = f"priority-{match['priority'].lower()}"
    
    st.markdown(f"""
    <div class="match-card {priority_class}">
        <h3>🏆 {rank}위. {match['title']}</h3>
        <div style="display: flex; gap: 1rem; margin: 0.5rem 0;">
            <span style="background: #e3f2fd; padding: 0.25rem 0.75rem; border-radius: 20px;">
                📊 {match['match_score']:.1f}점
            </span>
            <span style="background: #f3e5f5; padding: 0.25rem 0.75rem; border-radius: 20px;">
                📂 {match['data_type']}
            </span>
            <span style="background: #fff3e0; padding: 0.25rem 0.75rem; border-radius: 20px;">
                🔥 {match['priority']}
            </span>
        </div>
        <p style="margin-top: 1rem; color: #666;">{match['summary'][:200]}...</p>
        <p style="color: #999; font-size: 0.9rem;">📅 마감: {match['deadline']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 상세 정보 접기/펼치기
    with st.expander("📋 상세 정보"):
        st.write("**매칭 이유:**")
        for reason in match['reasons']:
            st.write(f"- {reason}")
        
        if match['detail_url']:
            st.markdown(f"[🔗 상세 페이지 바로가기]({match['detail_url']})")

# 메인 헤더
st.markdown('<div class="main-header">🚀 AI 창업지원 매칭 시스템</div>', unsafe_allow_html=True)

# 사이드바 - 프로필 입력
with st.sidebar:
    st.header("👤 프로필 입력")
    
    name = st.text_input("이름", "김창업")
    age = st.slider("나이", 18, 60, 29)
    region = st.selectbox(
        "지역",
        ["서울", "부산", "경기", "인천", "대전", "대구", "광주", "울산", 
         "세종", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
    )
    business_stage = st.selectbox(
        "창업단계",
        ["예비창업자", "3년이하", "7년이하", "10년이하"]
    )
    business_field = st.text_input("사업분야", "카페·외식업")
    target_type = st.selectbox(
        "대상유형",
        ["청년", "여성", "일반", "중장년", "예비창업자"]
    )
    
    is_veteran = st.checkbox("참전유공자")
    is_disabled = st.checkbox("장애인")
    
    additional_context = st.text_area(
        "추가 설명",
        "카페 창업 준비 중",
        height=100
    )
    
    st.divider()
    
    # 실행 버튼
    if st.button("🎯 매칭 시작", type="primary"):
        if not init_orchestrator():
            st.stop()
        
        profile = UserProfile(
            name=name,
            age=age,
            region=region,
            business_stage=business_stage,
            business_field=business_field,
            target_type=target_type,
            is_veteran=is_veteran,
            is_disabled=is_disabled,
            additional_context=additional_context
        )
        
        with st.spinner("🔄 매칭 중... (처음 실행은 시간이 걸립니다)"):
            st.session_state.report = st.session_state.orchestrator.run(
                profile, 
                top_n=10,
                use_cache=True
            )
        
        st.success("✅ 매칭 완료!")
        st.rerun()

# 메인 영역 - 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["📊 매칭 결과", "💬 챗봇 상담", "📈 분석", "💾 내보내기"])

with tab1:
    if st.session_state.report:
        report = st.session_state.report
        
        # 요약 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 매칭", report['total_matches'])
        with col2:
            st.metric("고우선순위", report['by_priority'].get('HIGH', 0))
        with col3:
            st.metric("중우선순위", report['by_priority'].get('MEDIUM', 0))
        with col4:
            st.metric("저우선순위", report['by_priority'].get('LOW', 0))
        
        # LLM 요약
        if report.get('llm_summary'):
            st.info(f"🧠 **AI 분석:** {report['llm_summary']}")
        
        st.divider()
        
        # 필터
        col1, col2 = st.columns(2)
        with col1:
            priority_filter = st.multiselect(
                "우선순위 필터",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"]
            )
        with col2:
            type_filter = st.multiselect(
                "유형 필터",
                list(report['by_type'].keys()),
                default=list(report['by_type'].keys())
            )
        
        # 필터링된 결과
        filtered = [
            r for r in report['recommendations']
            if r['priority'] in priority_filter and r['data_type'] in type_filter
        ]
        
        st.subheader(f"🎯 추천 결과 ({len(filtered)}개)")
        
        for idx, match in enumerate(filtered, 1):
            display_match_card(match, idx)
    else:
        st.info("👈 왼쪽 사이드바에서 프로필을 입력하고 '매칭 시작'을 눌러주세요")

with tab2:
    st.subheader("💬 AI 상담 챗봇")
    
    if not st.session_state.orchestrator:
        st.warning("먼저 프로필을 입력하고 매칭을 실행해주세요")
    else:
        # 채팅 기록 표시
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg['user'])
            with st.chat_message("assistant"):
                st.write(msg['bot'])
        
        # 입력창
        if question := st.chat_input("질문을 입력하세요"):
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.write(question)
            
            # AI 응답 생성
            profile = UserProfile(
                name=name, age=age, region=region,
                business_stage=business_stage,
                business_field=business_field,
                target_type=target_type,
                is_veteran=is_veteran,
                is_disabled=is_disabled,
                additional_context=additional_context
            )
            
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    answer = st.session_state.orchestrator.chatbot.chat(
                        profile, question, st.session_state.chat_history
                    )
                st.write(answer)
            
            # 기록 저장
            st.session_state.chat_history.append({
                'user': question,
                'bot': answer
            })

with tab3:
    if st.session_state.report:
        report = st.session_state.report
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 점수 분포")
            scores = [r['match_score'] for r in report['recommendations']]
            st.bar_chart(pd.DataFrame({'점수': scores}))
        
        with col2:
            st.subheader("📂 유형별 분포")
            type_df = pd.DataFrame(
                list(report['by_type'].items()),
                columns=['유형', '개수']
            )
            st.bar_chart(type_df.set_index('유형'))
    else:
        st.info("매칭 결과가 없습니다")

with tab4:
    if st.session_state.report:
        report = st.session_state.report
        
        st.subheader("💾 결과 내보내기")
        
        # JSON 다운로드
        col1, col2 = st.columns(2)
        with col1:
            safe_report = to_serializable(report)  # 🔹 추가
            json_str = json.dumps(safe_report, ensure_ascii=False, indent=2)
            st.download_button(
                label="📄 JSON 다운로드",
                data=json_str,
                file_name=f"matching_{report['profile']['name']}.json",
                mime="application/json"
            )

        
        # CSV 다운로드
        with col2:
            df = pd.DataFrame([
                {
                    '순위': r['rank'],
                    '제목': r['title'],
                    '점수': r['match_score'],
                    '우선순위': r['priority'],
                    '유형': r['data_type'],
                    '마감일': r['deadline']
                }
                for r in report['recommendations']
            ])
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 CSV 다운로드",
                data=csv,
                file_name=f"matching_{report['profile']['name']}.csv",
                mime="text/csv"
            )
    else:
        st.info("내보낼 결과가 없습니다")
