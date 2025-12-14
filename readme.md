# 🤖 Agentic AI 기반 창업지원 매칭 시스템

## 🎯 프로젝트 개요

**자율적으로 판단하고 행동하는 Agentic AI**를 활용한 맞춤형 창업지원 추천 시스템입니다.

### ✨ 주요 특징

- 🧠 **Agentic AI**: 에이전트가 스스로 판단하고 도구를 선택
- 🔧 **Tool System**: 검색, 필터링, 순위 매기기 등 독립적인 도구들
- 💬 **대화형 챗봇**: 카카오톡 스타일 UI
- 📊 **RAG 기반 검색**: 벡터 데이터베이스 활용
- 🎨 **실시간 추천**: 사용자 프로필 맞춤형

---

## 📋 시스템 요구사항

- Python 3.8+
- 4GB+ RAM
- 인터넷 연결 (API 호출용)

---

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd startup-support-agentic

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 설정

`.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 파일 편집:

```env
# 필수
KISED_SERVICE_KEY=your_kised_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# 선택
LLM_MODEL=gpt-4o-mini
DATA_DAYS_RANGE=180
CACHE_REFRESH_DAYS=1
```

### 3. 실행

#### 웹 인터페이스 (Streamlit)

```bash
streamlit run web/streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

#### CLI 테스트

```bash
# 전체 테스트
python main.py all

# 추천 시스템만
python main.py recommend

# 챗봇만
python main.py chat

# 대화형 모드
python main.py interactive
```

---

## 🏗️ 아키텍처

### Agentic AI 작동 방식

```
User Input
    ↓
[Orchestrator]
    ↓
[Agentic Agent]
    ↓
Reasoning Loop:
    1. 상황 분석 (LLM)
    2. 도구 선택
    3. 도구 실행
    4. 결과 해석
    5. 다음 행동 결정
    ↓
    반복 or 종료
    ↓
Final Result
```

### 도구 시스템

| 도구 | 설명 | 사용 시점 |
|------|------|-----------|
| `search_database` | 벡터 검색 | 정보가 필요할 때 |
| `filter_results` | 조건 필터링 | 검색 결과가 많을 때 |
| `rank_results` | 순위 매기기 | 우선순위 결정 필요 시 |
| `analyze_match` | 적합성 분석 | 상세 이유 필요 시 |
| `summarize_recommendations` | 전체 요약 | 최종 결과 생성 시 |

---

## 📁 프로젝트 구조

```
startup-support-agentic/
├── config/              # 설정
│   └── settings.py
├── core/                # 핵심 시스템
│   ├── orchestrator.py  # Agentic Orchestrator
│   ├── tools.py         # Tool System
│   ├── rag_system.py    # RAG Engine
│   └── llm_client.py    # LLM Client
├── agents/              # 에이전트들
│   ├── agentic_base.py  # Agentic Agent Base
│   ├── data_collector.py
│   └── rag_builder.py
├── models/              # 데이터 모델
├── utils/               # 유틸리티
└── web/                 # 웹 인터페이스
    └── streamlit_app.py
```

---

## 🔧 주요 기능

### 1. 맞춤형 추천

```python
from core.orchestrator import AgenticOrchestrator
from models.data import UserProfile

orchestrator = AgenticOrchestrator(
    service_key="your_key",
    llm_api_key="your_key"
)

profile = UserProfile(
    age=29,
    region="서울",
    business_stage="예비창업자",
    business_field="AI",
    target_type="청년"
)

report = orchestrator.run(profile, top_n=10)
```

### 2. 대화형 챗봇

```python
answer = orchestrator.chat(
    profile=profile,
    question="서울 청년 지원 프로그램 알려줘",
    category="지원사업/공고"
)
```

---

## 🎨 웹 인터페이스

### 카카오톡 스타일 챗봇

- 노란색 말풍선 (사용자)
- 흰색 말풍선 (봇)
- 실시간 타임스탬프
- 빠른 답변 버튼
- 대화 히스토리

### 맞춤형 추천

- 프로필 기반 필터링
- 마감 공고 자동 제외
- 실시간 점수 계산
- 상세 정보 링크

---

## 🔍 데이터 소스

### K-Startup API

- 지원사업 공고
- 통합 사업 정보
- 자료실 콘텐츠
- 통계자료

### 창업에듀 API

- 온라인 강좌
- 교육 프로그램

### 창업공간 API

- 창업공간 정보
- 센터 정보

### 창업기업 확인서 API

- 인증 제품
- 인증 기업

---

## ⚙️ 설정 옵션

### 환경 변수

```env
# API Keys
KISED_SERVICE_KEY=           # K-Startup API 키
OPENAI_API_KEY=              # OpenAI API 키

# LLM Settings
LLM_PROVIDER=openai          # openai 또는 anthropic
LLM_MODEL=gpt-4o-mini        # 사용할 모델

# Data Collection
DATA_DAYS_RANGE=180          # 공고 수집 범위 (일)
MAX_PAGES_PER_ENDPOINT=5     # 페이지당 수집량

# Cache
CACHE_REFRESH_DAYS=1         # 캐시 갱신 주기

# Web
WEB_PORT=8501                # Streamlit 포트

# Logging
AGENT_VERBOSE=false          # 에이전트 로그 출력
LOG_LEVEL=INFO               # 로그 레벨
```

---

## 🧪 테스트

### 단위 테스트

```bash
pytest tests/
```

### 통합 테스트

```bash
python main.py all
```

---

## 🐛 문제 해결

### API 키 오류

```
❌ 설정 오류: KISED_SERVICE_KEY 환경변수가 설정되지 않았습니다
```

**해결**: `.env` 파일에 API 키 추가

### 메모리 부족

```
❌ MemoryError: Unable to allocate array
```

**해결**: `MAX_PAGES_PER_ENDPOINT` 값 줄이기

### LLM 오류

```
❌ LLM 오류: rate_limit_exceeded
```

**해결**: API 요청 제한 확인, 잠시 후 재시도

---

## 📊 성능 최적화

### 캐시 활용

- RAG 인덱스: 1일 캐시
- 원시 데이터: 1일 캐시
- 자동 갱신

### 메모리 관리

- 문서 청킹
- 임베딩 배치 처리
- 가비지 컬렉션

---

## 🔐 보안

- API 키는 환경변수로 관리
- `.env` 파일은 `.gitignore`에 추가
- 사용자 데이터 로컬 저장

---

## 📝 라이선스

MIT License

---

## 👥 기여

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📧 문의

- 이슈: GitHub Issues
- 이메일: your@email.com

---

## 🙏 감사의 말

- K-Startup Open API
- OpenAI
- Streamlit
- Sentence Transformers

---

**Made with ❤️ by Agentic AI Team**