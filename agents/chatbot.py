"""
챗봇 에이전트
"""
from typing import List, Dict, Optional, Tuple, Any
import json

from agents.base import AgenticAgent
from models.data import UserProfile, Document, QuestionPlan
from utils.text import safe_get

# Streamlit에서 선택한 카테고리 → RAG 문서 type 매핑
CATEGORY_TYPE_MAP = {
    "지원사업/공고": ["business", "announcement"],
    "교육/강좌": ["lecture", "content"],
    "창업공간/센터": ["space", "center"],
    "인증 제품/기업": ["product", "corporate"],
    "기관/통계 자료": ["institution", "statistical"],
}


class ChatbotAgent(AgenticAgent):
    """대화형 챗봇 에이전트"""

    def __init__(self, rag_system: object, llm_client: Optional[object] = None):
        super().__init__("Chatbot", llm_client)
        self.rag = rag_system
        self.last_retrieved: List[Tuple[Document, float]] = []
        self.last_filters: Dict[str, str] = {}

    # ───────────────── 질문 플래너 ─────────────────
    def _plan_question(
        self,
        profile: UserProfile,
        question: str,
        category: Optional[str] = None,
    ) -> QuestionPlan:
        """
        LLM으로 질문 의도 + 사용할 데이터 타입을 먼저 계획하는 작은 에이전트
        """
        # LLM이 아예 없으면 안전한 기본값
        if not self.llm:
            if category and category in CATEGORY_TYPE_MAP:
                return QuestionPlan(
                    intents=["RECOMMEND"],
                    data_types=CATEGORY_TYPE_MAP[category],
                    answer_style="recommend",
                )
            # 기본: 지원사업/공고 위주
            return QuestionPlan(
                intents=["RECOMMEND"],
                data_types=["announcement", "business"],
                answer_style="recommend",
            )

        category_hint = ""
        if category and category in CATEGORY_TYPE_MAP:
            category_hint = (
                f"UI에서 사용자가 '{category}' 카테고리를 선택했습니다. "
                f"이 경우 우선적으로 {CATEGORY_TYPE_MAP[category]} 타입 데이터를 고려하세요."
            )

        plan_prompt = f"""
너는 '질문 플래너' 역할이다.
아래 사용자 정보와 질문을 보고, 어떤 자료를 찾아야 할지 계획을 JSON으로만 출력하라.

[사용자 정보]
- 나이: {profile.age}
- 지역: {profile.region}
- 창업 단계: {profile.business_stage}
- 분야: {profile.business_field}
- 대상 유형: {profile.target_type}

[질문]
{question}

[가능한 intent 종류]
- "RECOMMEND": 어떤 지원사업/강좌/공간 등을 추천해달라는 질문
- "REQUIRED_DOCS": 필요한 제출서류, 준비물, 증빙자료를 물어보는 질문
- "ELIGIBILITY": 신청 자격, 나이/기간/대상 조건을 물어보는 질문
- "DEADLINE": 신청 기간, 마감일을 물어보는 질문
- "PROCESS": 신청 방법, 절차, 어디서 신청하는지 물어보는 질문
- "SPACE_INFO": 창업공간/센터에 대한 정보를 찾는 질문
- "COURSE_INFO": 교육/강좌/온라인 강의에 대한 질문
- "GENERAL_QA": 그냥 설명이나 조언을 구하는 일반 질문

[데이터 타입(type) 종류]
- "announcement": 지원사업 공고
- "business": 통합 사업 정보
- "lecture": 창업에듀 강좌
- "content": 자료실 콘텐츠
- "space": 창업공간
- "center": 센터 정보
- "product": 인증 제품
- "corporate": 인증 기업
- "institution": 기관 정보
- "statistical": 통계/보고서

{category_hint}

규칙:
1. intents는 질문에 맞는 것 1~3개까지 고르고, 중요도 순서대로 적어라.
2. data_types는 이 질문에 답할 때 RAG에서 우선 조회해야 할 type을 1~4개 정도 골라라.
3. answer_style은 아래 중 하나만 골라라:
   - "recommend": 추천 위주 (몇 개 골라서 소개)
   - "explain": 조건/서류/절차 등 설명 위주
   - "mixed": 추천 + 설명을 같이

출력 형식 (반드시 아래 JSON 형식만, 다른 말 X):

{{
  "intents": ["RECOMMEND", "DEADLINE"],
  "data_types": ["announcement", "business"],
  "answer_style": "mixed"
}}
        """.strip()

        try:
            raw = self.llm.generate(plan_prompt, "질문 플래너", max_tokens=300)
            text = str(raw).strip()

            # 혹시 코드블럭으로 감싸져 있으면 제거
            if "```" in text:
                parts = text.split("```")
                # 마지막 조각이 JSON일 가능성이 제일 높음
                text = parts[-1].strip()

            data = json.loads(text)
            intents = data.get("intents", [])
            data_types = data.get("data_types", [])
            answer_style = data.get("answer_style", "auto")
            return QuestionPlan(
                intents=intents or ["GENERAL_QA"],
                data_types=data_types or ["announcement", "business"],
                answer_style=answer_style or "auto",
            )
        except Exception as e:
            # 실패하면 기본값으로 폴백
            self.think(
                "질문 계획 생성 실패",
                action="plan_question",
                result=str(e),
                confidence=0.1,
            )
            if category and category in CATEGORY_TYPE_MAP:
                return QuestionPlan(
                    intents=["RECOMMEND"],
                    data_types=CATEGORY_TYPE_MAP[category],
                    answer_style="recommend",
                )
            return QuestionPlan(
                intents=["GENERAL_QA"],
                data_types=["announcement", "business"],
                answer_style="auto",
            )

    # ───────────────── 실제 챗봇 ─────────────────
    def chat(
        self,
        profile: UserProfile,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        overrides: Optional[Dict[str, str]] = None,
        category: Optional[str] = None,
        faq_id: Optional[str] = None,
    ) -> str:
        # 프로필 None 방지
        if profile is None:
            profile = UserProfile(
                name="사용자",
                age=0,
                region="전국",
                business_stage="전채",
                business_field="전체",
                target_type="전체",
                additional_context="",
            )

        question = question.strip()
        if not question:
            return "질문을 입력해주세요 "

        overrides = overrides or {}

        # 컨텍스트용 프로필 값
        ctx_region = overrides.get("region", profile.region or "전국")
        ctx_field = overrides.get("business_field", profile.business_field or "전체")
        ctx_stage = overrides.get("business_stage", profile.business_stage or "전체")
        ctx_target = overrides.get("target_type", profile.target_type or "전체")

        # 로그용
        self.last_filters = {
            "region": ctx_region,
            "business_field": ctx_field,
            "business_stage": ctx_stage,
            "target_type": ctx_target,
            "faq_id": faq_id or "",
            "category": category or "",
        }

        # 1) 먼저 질문 계획 세우기 (agentic planner)
        plan = self._plan_question(profile, question, category=category)

        # 2) RAG 검색용 쿼리 만들기
        intent_hint = ", ".join(plan.intents)
        query_parts = [
            f"질문: {question}",
            f"질문 의도: {intent_hint}",
            f"지역: {ctx_region}",
            f"사업단계: {ctx_stage}",
            f"사업분야: {ctx_field}",
            f"대상: {ctx_target}",
        ]
        query = " / ".join([p for p in query_parts if p])

        # 3) RAG에서 문서 검색
        retrieved = self.rag.retrieve(query, top_k=top_k * 3)
        self.last_retrieved = retrieved

        # 4) 플래너가 정한 data_types 기준으로 필터링
        type_whitelist: List[str] = plan.data_types[:]

        # 플래너가 아무것도 안 줬고, UI 카테고리가 있으면 참고
        if not type_whitelist and category and category in CATEGORY_TYPE_MAP:
            type_whitelist = CATEGORY_TYPE_MAP[category]

        filtered_docs: List[Tuple[Document, float]] = []
        for doc, score in retrieved:
            meta = doc.metadata or {}
            doc_type = str(meta.get("type", ""))

            # 화이트리스트가 없으면 다 사용 / 있으면 딱 맞는 타입만 사용
            if (not type_whitelist) or any(t == doc_type for t in type_whitelist):
                filtered_docs.append((doc, score))

        # 너무 많이 걸러졌으면 원본 상위 N개로 폴백
        if not filtered_docs:
            filtered_docs = retrieved[:top_k]
        else:
            filtered_docs = filtered_docs[:top_k]

        # 5) 컨텍스트 텍스트 합치기
        context_lines: List[str] = []
        for i, (doc, score) in enumerate(filtered_docs, start=1):
            meta = doc.metadata or {}
            title = meta.get("title", f"문서 {i}")
            context_lines.append(
                f"[{i}] {title} (type={meta.get('type','')}, score={score:.2f})"
            )
            context_lines.append(doc.text[:500])
            context_lines.append("")

        context_text = "\n".join(context_lines) if context_lines else ""

        # 6) LLM이 없으면 컨텍스트만 반환
        if not self.llm:
            if not context_text:
                return "관련된 정보를 찾지 못했습니다. 질문을 조금 더 구체적으로 적어주세요."
            return (
                "아직 LLM이 연결되지 않아서, 관련 자료만 정리해서 보여드릴게요.\n\n"
                + context_text
            )

        # ── plan에 따라 답변 모드 설명 만들기 ──
        intent_desc_lines: List[str] = []

        if "REQUIRED_DOCS" in plan.intents:
            intent_desc_lines.append(
                "- 필요한 제출서류, 증빙자료, 준비물을 우선 정리해서 알려주세요."
            )
        if "ELIGIBILITY" in plan.intents:
            intent_desc_lines.append(
                "- 나이, 창업기간, 대상 유형 등 신청 자격 조건을 표나 목록으로 정리해주세요."
            )
        if "DEADLINE" in plan.intents:
            intent_desc_lines.append(
                "- 신청 시작일, 마감일, 접수시간 등 기간 정보를 명확하게 써 주세요."
            )
        if "PROCESS" in plan.intents:
            intent_desc_lines.append(
                "- 어디에서 어떤 순서로 신청하면 되는지 절차를 단계별로 설명해주세요."
            )
        if "SPACE_INFO" in plan.intents:
            intent_desc_lines.append(
                "- 창업공간/센터의 위치, 특징, 이용방법을 중심으로 설명해주세요."
            )
        if "COURSE_INFO" in plan.intents:
            intent_desc_lines.append(
                "- 온라인/오프라인 교육·강좌 정보를 중심으로 소개해주세요."
            )
        if "RECOMMEND" in plan.intents:
            intent_desc_lines.append(
                "- 사용자의 나이·지역·단계·분야에 맞는 지원사업/강좌/공간을 2~3개 골라 추천해주세요."
            )
        if not intent_desc_lines:
            intent_desc_lines.append(
                "- 질문 내용에 맞게 필요한 정보(조건, 기간, 서류, 추천 등)를 자연스럽게 설명해주세요."
            )

        mode_text = "\n".join(intent_desc_lines)

        # answer_style에 따라 말투/구성 힌트
        style_hint = ""
        if plan.answer_style == "recommend":
            style_hint = "추천 항목마다 한두 문장 설명과 '왜 도움이 되는지' 이유를 꼭 적어주세요."
        elif plan.answer_style == "explain":
            style_hint = "가능하면 항목별 목록을 사용해서 보기 쉽게 정리해주세요."
        else:  # mixed / auto
            style_hint = (
                "핵심 정보(조건·기간·서류)를 먼저 정리하고, 필요하면 관련 지원사업을 예로 들어주세요."
            )

        prompt = f"""
당신은 한국의 청년/창업자를 돕는 친절한 상담 챗봇입니다.

[사용자 정보]
- 나이: {profile.age}세
- 지역: {ctx_region}
- 창업 단계: {ctx_stage}
- 창업/관심 분야: {ctx_field}
- 대상 유형: {ctx_target}

[사용자 질문]
{question}

[관련 지원사업/콘텐츠/공간 정보]
{context_text}

[답변 모드]
{mode_text}
- {style_hint}
- 전체 길이는 3~8문장 정도로 설명하세요.
- "위 문서에 따르면" 같은 말보다는 실제 상담사가 말하듯 자연스럽게 답변하세요.
        """.strip()

        system_msg = "창업지원 전문 상담사처럼, 친절하고 구체적으로 답변해 주세요."

        try:
            raw = self.llm.generate(prompt, system_msg, max_tokens=700)
            return str(raw).strip()
        except Exception as e:
            self.think(
                "챗봇 답변 생성 중 오류",
                action="llm.generate",
                result=str(e),
                confidence=0.1,
            )
            # LLM이 터져도 컨텍스트는 보여주기
            if context_text:
                return (
                    "현재 상담 모듈에서 오류가 발생했지만, 관련 자료는 아래에 정리해 드릴게요.\n\n"
                    + context_text
                )
            return "현재 상담 기능에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
