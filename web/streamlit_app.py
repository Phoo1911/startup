# web/streamlit_app.py
import json
from pathlib import Path
import sys
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# 프로젝트 루트 경로 (web 폴더의 한 단계 위)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from config.settings import Config
from models.data import UserProfile
from core.orchestrator import AgenticOrchestrator


# ───────────────── 기본 설정 ─────────────────
st.set_page_config(
    page_title="AI 창업지원 시스템",
    layout="wide",
)

# ───────────────── 세션 초기화 ─────────────────
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgenticOrchestrator(
        service_key=Config.SERVICE_KEY,
        llm_api_key=Config.LLM_API_KEY,
    )

if "report" not in st.session_state:
    st.session_state.report = None

if "profile" not in st.session_state:
    st.session_state.profile = None

# 채팅 히스토리: [{"role": "user"|"assistant", "content": "..."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ✅ UI 라벨 ↔ 내부 data_type 매핑
DATA_TYPE_LABELS = {
    # kisedKstartupService01
    "지원사업 공고": "announcement",
    "통합공고(사업정보)": "business",
    "자료실 콘텐츠": "content",
    "통계자료": "statistical",
    # kisedEduService
    "교육·강좌": "lecture",
    # kisedSlpService
    "창업공간": "space",
    "창업센터": "center",
    # kisedCertService
    "인증 제품": "product",
    "인증 기업": "corporate",
    # kisedInsttInfoService
    "지원기관": "institution",
}

# ───────────────── 사이드바: 프로필 입력 ─────────────────
st.sidebar.title("👤 사용자 정보 입력")

with st.sidebar.form("profile_form"):
    age = st.number_input("나이", min_value=18, max_value=80, value=29, step=1)
    region = st.selectbox(
        "지역",
        [
            "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
            "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남",
            "제주", "전국",
        ],
        index=0,
    )
    stage = st.selectbox(
        "창업단계",
        ["예비창업자", "1년미만", "2년미만", "3년미만", "5년미만", "7년미만"],
        index=0,
    )
    field = st.text_input("사업분야 (예: AI, 제조, 콘텐츠 등)", value="AI")
    target = st.selectbox(
        "대상유형",
        ["청소년", "대학생", "일반인"],
        index=0,
    )

    veteran = st.checkbox("참전유공자 여부")
    disabled = st.checkbox("장애인 여부")
    context = st.text_area(
        "추가 설명 (선택)",
        placeholder="관심있는 지원사업, 희망 프로그램 등 자유 입력",
    )

    # 조회하고 싶은 데이터 유형 선택
    selected_labels = st.multiselect(
        "조회하고 싶은 데이터 유형",
        options=list(DATA_TYPE_LABELS.keys()),
        default=["지원사업 공고", "통합공고(사업정보)"],
    )

    top_n = st.slider("추천 개수", min_value=5, max_value=30, value=10, step=1)
    use_cache = st.checkbox("캐시 사용 ", value=True)

    run_match = st.form_submit_button("매칭 실행")

# ───────────────── 메인 레이아웃 ─────────────────
st.title(" AI 창업지원 매칭 & 상담 시스템")

tab_result, tab_chat = st.tabs([" 추천 결과", " 상담 챗봇"])

# ───────────────── 매칭 실행 ─────────────────
# ───────────────── 매칭 실행 ─────────────────
if run_match:
    desired_types = [DATA_TYPE_LABELS[l] for l in selected_labels]

    profile = UserProfile(
        name="사용자",
        age=age,
        region=region,
        business_stage=stage,
        business_field=field,
        target_type=target,
        is_veteran=veteran,
        is_disabled=disabled,
        company_name=None,
        additional_context=context,
        desired_data_types=desired_types,
    )

    st.session_state.profile = profile

    # 🔥 여기서는 "Semantic 기반 Hybrid" 사용
    with st.spinner("🤖 AI가 최적의 지원사업을 찾고 있습니다..."):
        print("\n" + "="*80)
        print("🚀 Semantic Hybrid 추천 시작")
        print(f"  - 지역: {region}")
        print(f"  - 분야: {field}")
        print(f"  - 단계: {stage}")
        print(f"  - 데이터 타입: {desired_types}")
        print("="*80)

        # ✅ 여기서 run_agentic → run 으로 변경
        st.session_state.report = st.session_state.orchestrator.run(
            profile=profile,
            top_n=top_n,
            use_cache=use_cache,
            base_query="최신 창업지원 관련 정보 중심으로",
        )

        rec_count = len(st.session_state.report.get("recommendations", []))
        if rec_count > 0:
            st.success(f"✅ {rec_count}개의 추천 결과를 찾았습니다!")
        else:
            st.warning("⚠️ 조건에 맞는 결과가 없습니다. 터미널의 디버그 로그를 확인해주세요.")


# ───────────────── 탭 1: 추천 결과 ─────────────────
with tab_result:
    st.header(" 추천 결과")

    report = st.session_state.report

    if not report:
        st.info("👈 왼쪽에서 정보를 입력하고 **[매칭 실행]** 버튼을 눌러주세요.")
    else:
        recommendations = report.get("recommendations", [])

        if not recommendations:
            st.warning("조건에 맞는 매칭 결과가 없습니다. 프로필 조건을 조금 넓혀보세요.")
        else:
            profile_dict = report.get("profile", {})

            st.subheader("👤 내 프로필 기준 추천 결과")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("지역", profile_dict.get('region', '-'))
            with col2:
                st.metric("창업단계", profile_dict.get('business_stage', '-'))
            with col3:
                st.metric("사업분야", profile_dict.get('business_field', '-'))
            with col4:
                st.metric("대상유형", profile_dict.get('target_type', '-'))

            st.divider()

            st.subheader("🎯 상위 추천 리스트")

            all_recs = recommendations

            profile_obj = st.session_state.profile

            # 사용자가 고른 타입 (예: ["announcement", "business"])
            desired_types = set(getattr(profile_obj, "desired_data_types", []) or [])

            # 실제 추천 결과 안에 들어있는 타입들
            available_types = {rec.get("data_type", "기타") for rec in all_recs}

            # ✅ 교집합이 있으면 그 교집합만, 없으면 일단 전부 보여주기
            if desired_types:
                overlap = desired_types & available_types
                if overlap:
                    desired_types = overlap
                else:
                    desired_types = available_types  # 선택한 타입이 하나도 없으면 그냥 다 보여줌
            else:
                desired_types = available_types

            st.caption("📌 조회 데이터 유형: " + ", ".join(selected_labels))

            filtered_recs = []
            for rec in all_recs:
                data_type = rec.get("data_type", "기타")
                if data_type not in desired_types:
                    continue
                filtered_recs.append(rec)

            if not filtered_recs:
                st.info(
                    "조건에 맞는 추천 결과가 없습니다. 조회 데이터 유형이나 지역 조건을 넓혀보세요."
                )
            else:
                # 🔥 중복 제거 (title + region 기반)
                seen_keys = set()
                deduplicated = []
                for rec in filtered_recs:
                    # ID가 있으면 ID 사용, 없으면 title+region 조합
                    rec_id = rec.get("id")
                    title = rec.get("title", "")
                    region = rec.get("region") or rec.get("metadata", {}).get("region") or ""
                    
                    if rec_id:
                        key = rec_id
                    else:
                        key = f"{title}_{region}"
                    
                    if key in seen_keys:
                        continue
                    
                    seen_keys.add(key)
                    deduplicated.append(rec)

                for idx, rec in enumerate(deduplicated, start=1):
                    data_type = rec.get("data_type", "기타")
                    
                    # 카드 스타일로 표시
                    with st.container():
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            rank = rec.get("rank", idx)
                            title = rec.get("title", "제목 없음")
                            st.markdown(f"### {rank}. {title}")

                            # 메타 정보
                            meta_line = []
                            region = rec.get("region") or rec.get("metadata", {}).get("region")
                            field = rec.get("field") or rec.get("metadata", {}).get("field")
                            
                            if region:
                                meta_line.append(f"📍 {region}")
                            if field:
                                meta_line.append(f"💼 {field}")

                            host_org = rec.get("host_org") or rec.get("metadata", {}).get("host_org")
                            if host_org:
                                lower = host_org.lower()
                                if "_tab" not in lower and not lower.endswith("_tab1"):
                                    meta_line.append(f"🏢 {host_org}")

                            if meta_line:
                                st.write(" | ".join(meta_line))

                            # 매칭 우선순위
                            priority = rec.get("priority", "기타")
                            st.write(f"(**{priority}** 우선순위)")

                            # 🔥 접수기간/마감일 정보 가져오기
                            meta = rec.get("metadata", {})
                            extra = rec.get("extra", {})
                            
                            apply_period = (
                                rec.get("apply_period") or 
                                extra.get("apply_period") or 
                                meta.get("apply_period") or 
                                ""
                            )
                            
                            deadline = (
                                rec.get("deadline") or 
                                extra.get("deadline") or 
                                meta.get("deadline") or 
                                ""
                            )

                            # 데이터 타입별 정보 표시
                            if data_type in ("announcement", "business"):
                                # 🔥 접수기간 표시 로직 개선
                                if apply_period:
                                    st.write(f"📅 **접수기간:** {apply_period}")
                                elif deadline:
                                    st.write(f"📅 **접수기간:**  {deadline}")
                                else:
                                    st.write("📅 접수기간: 정보 없음")

                                # 🔥 마감 임박 경고
                                if deadline:
                                    try:
                                        deadline_clean = deadline.replace("-", "").replace(".", "")
                                        if len(deadline_clean) >= 8 and deadline_clean[:8].isdigit():
                                            from datetime import datetime
                                            deadline_date = datetime.strptime(deadline_clean[:8], "%Y%m%d").date()
                                            today = datetime.now().date()
                                            days_left = (deadline_date - today).days
                                            
                                            if days_left < 0:
                                                st.error(f"⚠️ 마감됨 ({abs(days_left)}일 전)")
                                            elif 0 <= days_left <= 3:
                                                st.error(f"🔥 마감 임박! D-{days_left}")
                                            elif 0 <= days_left <= 7:
                                                st.warning(f"⏰ D-{days_left}")
                                    except Exception:
                                        pass

                            elif data_type == "lecture":
                                reg_date = rec.get("reg_date") or meta.get("reg_date") or extra.get("reg_date") or ""
                                view_cnt = rec.get("view_cnt") or meta.get("view_cnt") or extra.get("view_cnt") or ""
                                play_time = rec.get("play_time") or meta.get("play_time") or extra.get("play_time") or 0
                                
                                info_parts = []
                                if reg_date:
                                    info_parts.append(f"📅 등록일: {reg_date}")
                                if view_cnt not in ("", None):
                                    info_parts.append(f"👁 조회수: {view_cnt}")
                                if play_time and int(play_time) > 0:
                                    mins = int(play_time) // 60
                                    info_parts.append(f"⏱ 재생시간: {mins}분")
                                
                                if info_parts:
                                    st.write(" / ".join(info_parts))

                            elif data_type in ("product", "corporate"):
                                if deadline:
                                    st.write(f"📅 **인증 유효기간:**  {deadline}")

                            elif data_type in ("space", "center"):
                                address = rec.get("address") or meta.get("address") or extra.get("address") or ""
                                rent = rec.get("rent") or meta.get("rent") or extra.get("rent") or 0
                                seat_count = rec.get("seat_count") or meta.get("seat_count") or extra.get("seat_count") or ""
                                
                                info_parts = []
                                if address:
                                    info_parts.append(f"📍 {address}")
                                if rent and int(rent) > 0:
                                    info_parts.append(f"💰 임대료: {rent:,}원")
                                if seat_count:
                                    info_parts.append(f"🪑 좌석: {seat_count}")
                                
                                if info_parts:
                                    st.write(" | ".join(info_parts))

                            else:
                                if deadline:
                                    st.write(f"📅 **유효기간:**  {deadline}")

                            # 매칭 이유
                            reasons = rec.get("reasons") or rec.get("match_reasons") or []
                            if reasons:
                                st.write("🔎 **매칭 이유:**")
                                for reason in reasons[:3]:
                                    st.write(f"  • {reason}")

                            # 요약
                            summary = rec.get("summary")
                            if summary:
                                with st.expander("📄 내용 요약 보기"):
                                    st.write(summary)

                        with col2:
                            st.markdown("#### 🔗 관련 링크")

                            # 🔥 URL 가져오기 (여러 곳에서 시도)
                            detail_url = (
                                rec.get("detail_url") or 
                                meta.get("detail_url") or 
                                extra.get("detail_url") or 
                                ""
                            )
                            
                            guide_url = (
                                rec.get("guide_url") or 
                                meta.get("guide_url") or 
                                extra.get("guide_url") or 
                                ""
                            )
                            
                            apply_url = (
                                rec.get("apply_url") or 
                                meta.get("apply_url") or 
                                extra.get("apply_url") or 
                                ""
                            )

                            # detail_url 있을 때만 노출
                            if detail_url:
                                if data_type == "lecture":
                                    st.markdown(f"🎓 [강좌 보러가기]({detail_url})")
                                elif data_type in ("space", "center"):
                                    st.markdown(f"🏢 [공간 상세보기]({detail_url})")
                                elif data_type in ("product", "corporate"):
                                    st.markdown(f"🏭 [제품/기업 정보]({detail_url})")
                                else:
                                    st.markdown(f"📋 [상세 페이지]({detail_url})")

                            # 공고/사업만 안내문/신청 버튼
                            if data_type in ("announcement", "business"):
                                if guide_url:
                                    st.markdown(f"📑 [공고 안내문]({guide_url})")
                                if apply_url:
                                    st.markdown(f"✍️ [신청 바로가기]({apply_url})")

                        st.divider()

# ───────────────── 탭 2: 상담 챗봇 ─────────────────
with tab_chat:
    st.header("💬 창업지원 상담 챗봇")

    # 기본 프로필 (없으면 디폴트)
    profile = st.session_state.profile or UserProfile(
        name="",
        age=0,
        region="전국",
        business_stage="",
        business_field="",
        target_type="",
        is_veteran=False,
        is_disabled=False,
        additional_context="",
    )

    # 카테고리 선택
    category = st.radio(
        "궁금한 분야를 선택해주세요.",
        ["전체", "지원사업/공고", "교육/강좌", "창업공간/센터", "인증 제품/기업", "기관/통계 자료"],
        horizontal=True,
    )

    st.caption("💡 예시 버튼을 누르거나 아래 입력창에 직접 입력하세요!")

    # 예시 버튼
    quick_questions = [
        "부산 청년 지원 프로그램 알려줘",
        "온라인 창업 교육 뭐 있어?",
        "예비창업자 정부지원금 알려줘",
    ]
    cols = st.columns(len(quick_questions))
    for i, (col, q) in enumerate(zip(cols, quick_questions)):
        with col:
            if st.button(q, key=f"quick_{i}"):
                st.session_state["pending_input"] = q

    # 채팅 박스
    chat_box = st.container()

    # 입력창
    raw_input = st.chat_input("무엇이 궁금한가요?")
    user_prompt = st.session_state.pop("pending_input", None) or raw_input

    # 입력 처리
    if user_prompt:
        # 사용자 메시지 저장
        st.session_state.chat_history.append(
            {"role": "user", "content": user_prompt}
        )

        question_for_model = (
            f"[카테고리:{category}] {user_prompt}"
            if category != "전체"
            else user_prompt
        )

        try:
            with st.spinner(" 답변을 생성하고 있습니다..."):
                answer = st.session_state.orchestrator.chat(
                    profile=profile,
                    question=question_for_model,
                    history=st.session_state.chat_history,
                    top_k=5,
                    category=category,
                )
        except Exception as e:
            answer = f"⚠️ 오류 발생: {e}"

        # 어시스턴트 메시지 저장
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

    # 💬 채팅 출력
    with chat_box:
        # 오래된 것부터 차례대로
        for msg in st.session_state.chat_history[-40:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])