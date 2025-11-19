"""
챗봇 에이전트
"""
from typing import List, Dict, Optional, Tuple
from agents.base import AgenticAgent
from models.data import UserProfile, Document
from utils.text import safe_get

class ChatbotAgent(AgenticAgent):
    """대화형 챗봇 에이전트"""
    
    def __init__(self, rag_system: object, llm_client: Optional[object] = None):
        super().__init__("Chatbot", llm_client)
        self.rag = rag_system
        self.last_retrieved: List[Tuple[Document, float]] = []
    
    def chat(
        self,
        profile: UserProfile,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
    ) -> str:
        """사용자 질문에 대한 답변"""
        question = question.strip()
        if not question:
            return "질문을 입력해주세요 🙂"
        
        # 쿼리 생성
        base_query = (
            f"{question} 지역:{profile.region} "
            f"분야:{profile.business_field} "
            f"단계:{profile.business_stage} 대상:{profile.target_type}"
        )
        
        self.think("사용자 질문 처리", action=base_query[:60], confidence=0.9)
        
        # RAG 검색
        retrieved = self.rag.retrieve(base_query, top_k=top_k)
        self.last_retrieved = retrieved
        self.think("RAG 검색 완료", result=f"{len(retrieved)}개", confidence=0.95)
        
        if not retrieved:
            if not self.llm:
                return "지금은 관련 지원사업을 찾지 못했어요. 질문을 조금 더 자세히 써주실래요?"
            
            # LLM만으로 답변
            prompt = (
                f"사용자 질문: {question}\n"
                "한국의 창업지원 일반 상식을 바탕으로, 3문장 이내로 간단히 안내해주세요."
            )
            return self.llm.generate(prompt, "친절한 상담사처럼.", max_tokens=300).strip()
        
        # 검색 결과 정리
        context_blocks = []
        for (doc, sim) in retrieved:
            meta = doc.metadata
            title = meta.get("title", "제목없음")
            dtype = meta.get("type", "unknown")
            deadline = meta.get("deadline", "")
            detail_url = safe_get(meta.get("raw", {}), "detailUrl", default="")
            
            block = [
                f"[제목] {title}",
                f"[유형] {dtype}",
                f"[마감] {deadline}",
                f"[유사도] {sim:.2f}",
                f"[내용요약]\n{doc.text[:400]}",
            ]
            
            if detail_url:
                block.append(f"[URL] {detail_url}")
            
            context_blocks.append("\n".join(block))
        
        context_text = "\n\n---\n\n".join(context_blocks[:5])
        
        # LLM 없으면 RAG 결과만 반환
        if not self.llm:
            return (
                "다음 지원사업들이 질문과 가장 비슷해 보여요:\n\n"
                f"{context_text[:1000]}"
            )
        
        # 이전 대화 히스토리
        history_text = ""
        if history:
            for turn in history[-5:]:
                history_text += f"사용자: {turn['user']}\n챗봇: {turn['bot']}\n"
        
        # LLM 프롬프트
        system_prompt = (
            "너는 한국의 창업지원사업(정부·지자체·창업진흥원 등)을 설명해 주는 상담 챗봇이야.\n"
            "- 항상 이전 대화 흐름을 이어서 답해줘.\n"
            "- 사용자가 짧게 물으면, 바로 직전 답변을 계속 이어서 말해줘.\n"
            "- 이미 설명한 내용은 반복하지 말고, 필요한 부분만 짧게 연결해줘.\n"
            "- 모르면 모른다고 말하고, 근거 없는 내용은 추측하지 마.\n"
            "- 중학생도 이해할 수 있을 정도로 쉽게, 3~5문장 안으로 답해줘.\n"
            "- 가능하면 '1) 2) 3)' 구조는 유지하되, 첫 문장은 자연스럽게 연결해줘."
        )
        
        user_prompt = f"""
[사용자 프로필]
이름: {profile.name}
나이: {profile.age}
지역: {profile.region}
창업단계: {profile.business_stage}
사업분야: {profile.business_field}
대상유형: {profile.target_type}

[이전 대화]
{history_text if history_text else "없음"}

[사용자 현재 질문]
{question}

[검색된 공고 정보 (RAG 결과)]
{context_text}

위 정보를 참고해서,
1) 지금 질문에 대한 답을 앞 대화와 자연스럽게 이어서 설명하고,
2) 특히 질문에 나온 지역/조건이 있다면 그 기준으로 다시 정리해 주고,
3) 다음으로 사용자가 무엇을 하면 좋은지 한 줄 정도로 제안해 줘.
        """.strip()
        
        answer = self.llm.generate(user_prompt, system_prompt, max_tokens=500)
        return answer.strip()
