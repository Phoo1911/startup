# agents/llm_reasoner.py
"""
LLM 추론 에이전트
"""
from typing import List, Optional

from agents.base import AgenticAgent
from models.data import UserProfile, MatchResult


class LLMReasoningAgent(AgenticAgent):
    """LLM 기반 추론 에이전트"""

    def __init__(self, llm_client: Optional[object] = None):
        # llm_client는 .generate(prompt, system_msg, max_tokens=...) 메서드를 가진 객체라고 가정
        super().__init__("LLMReasoner", llm_client)

    def enhance_matches(self, matches: List[MatchResult], profile: UserProfile) -> List[MatchResult]:
        """
        매칭 결과 상위 N개에 대해
        → LLM이 '왜 적합한지' 2~3줄 이유를 전부 생성해서 match.reasons에 넣는다.
        """
        if not self.llm or not matches:
            return matches

       

        self.think(
            "LLM 분석 시작",
            action=f" 지원사업 자연어 이유 생성",
            confidence=0.85,
        )

        for match in matches:
            reason_list = self._analyze_with_llm_multi(match, profile)
            if reason_list:
                # 👉 룰 기반 reason은 버리고, 전부 LLM 이유로 덮어쓰기
                match.reasons = reason_list

        self.think("LLM 분석 완료", result="각 매칭에 자연어 이유 2~3줄 추가", confidence=0.9)
        return matches

    def _analyze_with_llm_multi(self, match: MatchResult, profile: UserProfile) -> List[str]:
        """
        한 매칭에 대해 LLM이 2~3줄 이유를 만들어 주도록 함.
        return: ["이유1", "이유2", ...]
        """
        age_text = f"{profile.age}세" if getattr(profile, "age", None) else "연령 정보 없음"
        region_text = profile.region or "지역 정보 없음"
        field_text = profile.business_field or "관심 분야 정보 없음"

        # 룰 기반에서 만든 힌트 (있으면)
        rule_reasons = []
        if match.extra and isinstance(match.extra, dict):
            rule_reasons = match.extra.get("rule_reasons", []) or []

        rule_text = ""
        if rule_reasons:
            bullet = "\n".join([f"- {r}" for r in rule_reasons])
            rule_text = f"\n[시스템이 참고한 포인트]\n{bullet}\n"
        else:
            rule_text = ""

        prompt = f"""
다음 사용자의 프로필과 지원사업 정보를 보고,
이 지원사업이 왜 잘 맞는지 2~3가지 이유를 한국어로 설명해 주세요.

[사용자 프로필]
- 나이: {age_text}
- 활동 지역: {region_text}
- 관심/업종 분야: {field_text}

[지원사업 정보]
- 제목: {match.title}
- 유형: {match.data_type}
- 지역: {match.region}
- 신청 기간: {match.apply_period}
- 요약/설명: {match.metadata.get("summary", "") or match.metadata.get("desc", "")}
{rule_text}

작성 규칙:
1. 번호를 붙여서 한 줄에 한 이유씩 써 주세요. (예: "1. ~", "2. ~").
2. 존댓말을 사용하세요.
3. 각 이유는 1문장 정도 길이로 써 주세요.
        """.strip()

        try:
            raw = self.llm.generate(
                prompt,
                "지시를 그대로 따르면서, 2~3줄의 이유만 써 주세요.",
                max_tokens=220,
            )
            text = str(raw).strip()

            # "1. ~\n2. ~" 형태를 라인별로 분리
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # 앞에 붙은 번호/불릿 제거
            cleaned: List[str] = []
            for ln in lines:
                if ln[0].isdigit() and "." in ln[:4]:
                    ln = ln.split(".", 1)[1].strip()
                elif ln.startswith(("-", "•", "·")):
                    ln = ln[1:].strip()
                if ln:
                    cleaned.append(ln)

            # 3줄까지만 사용
            return cleaned[:3]

        except Exception as e:
            self.think(
                "LLM 분석 중 오류",
                action=f"match_title={match.title}",
                result=str(e),
                confidence=0.2,
            )
            return []

    def generate_summary(self, matches: List[MatchResult], profile: UserProfile) -> str:
        """
        전체 추천 결과를 2~3문장으로 요약 + 격려 메시지 생성
        """
        if not matches:
            return f"{profile.name}님께 추천할 지원사업이 아직 없습니다."

        if not self.llm:
            return f"{profile.name}님께 {len(matches)}개 지원사업을 추천했습니다."

        top = matches[:3]
        matches_text = "\n".join([f"{i+1}. {m.title}" for i, m in enumerate(top)])

        prompt = f"""
다음 정보는 한 사용자를 위해 추천된 지원사업 목록입니다.

[사용자]
- 이름: {profile.name}
- 나이: {getattr(profile, "age", '')}
- 지역: {profile.region}
- 관심/업종 분야: {profile.business_field}

[추천 지원사업 상위 3개]
{matches_text}

위 정보를 보고,
1) 전체 추천 상황을 1~2문장으로 요약하고
2) 마지막에 짧은 응원 한 문장을 추가해 주세요.

조건:
- 존댓말을 사용하세요.
- 총 2~3문장 안에서 끝내 주세요.
        """.strip()

        try:
            raw = self.llm.generate(
                prompt,
                "친근한 상담사처럼, 쉬운 한국어로 2~3문장만 작성하세요.",
                max_tokens=200,
            )
            return str(raw).strip()
        except Exception as e:
            self.think(
                "LLM 요약 생성 중 오류",
                action="generate_summary",
                result=str(e),
                confidence=0.2,
            )
            return f"{profile.name}님께 적합한 지원사업을 여러 개 찾았습니다. 마음에 드는 사업을 하나씩 살펴보시면 좋겠습니다."
