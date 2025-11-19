# 🚀 AI 창업지원 매칭 시스템

Agentic AI + RAG + LLM 기반 창업지원사업 추천 시스템

## 📁 프로젝트 구조

```
startup-matching-system/
├── config/
│   └── settings.py              # 설정 관리
├── models/
│   ├── data_models.py           # 데이터 클래스
│   └── enums.py                 # Enum 정의
├── agents/                      # AI 에이전트들
│   ├── base_agent.py
│   ├── data_collector.py
│   ├── rag_builder.py
│   ├── semantic_matcher.py
│   ├── llm_reasoner.py
│   ├── recommender.py
│   └── chatbot.py
├── core/
│   ├── llm_client.py            # LLM 클라이언트
│   ├── rag_system.py            # RAG 시스템
│   └── orchestrator.py          # 전체 조율
├── utils/
│   ├── text_utils.py
│   └── date_utils.py
├── web/
│   ├── streamlit_app.py         # Streamlit UI (Option 1)
│   ├── fastapi_app.py           # FastAPI 서버 (Option 2)
│   ├── templates/               # HTML 템플릿
│   └── static/                  # CSS, JS
├── cache/                       # 데이터 캐시
├── reports/                     # 리포트 저장
├── requirements.txt
└── main.py                      # CLI 실행
```

## 🔧 설치

### 1. Python 환경 (3.9+)

```bash
# 가상환경 생성
python -m venv venv

# 활성화 (Windows)
venv\Scripts\activate

# 활성화 (Mac/Linux)
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정 (.env 파일)

```bash
# 필수
KISED_SERVICE_KEY=your_service_key_here

# LLM (선택사항)
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
LLM_MODEL=gpt-4o-mini

# 기타
AGENT_VERBOSE=false
EMBEDDING_MODEL=sentence-transformers
```

## 🚀 실행 방법

### Option 1: Streamlit (추천 - 빠른 프로토타입)

```bash
streamlit run web/streamlit_app.py
```

- 브라우저 자동 실행: http://localhost:8501
- 장점: 설치 후 바로 사용 가능
- 단점: 커스터마이징 제한

### Option 2: FastAPI (프로덕션)

```bash
uvicorn web.fastapi_app:app --reload
```

- 접속: http://localhost:8000
- API 문서: http://localhost:8000/docs
- 장점: 완전한 커스터마이징, REST API
- 단점: 프론트엔드 직접 구현 필요

### Option 3: CLI

```bash
python main.py --name "김창업" --region "서울" --field "AI"
```

## 📊 주요 기능

### 1. 프로필 기반 매칭
- 나이, 지역, 사업분야, 창업단계 등 고려
- AI 의미 분석으로 정확한 매칭
- 우선순위 자동 계산

### 2. RAG (검색증강생성)
- 실시간 공공데이터 수집
- Sentence-BERT 임베딩
- 유사도 기반 검색

### 3. LLM 추론 (선택)
- 개인화된 추천 이유 설명
- 챗봇 상담 기능
- GPT/Claude 지원

### 4. 데이터 소스
- 통합공고 지원사업 정보
- 지원사업 공고 정보
- 창업에듀 강좌
- 창업공간/센터
- 주관기관 정보

## 🎨 웹 UI 스크린샷

### Streamlit
- 간단한 사이드바 입력
- 실시간 결과 표시
- 인터랙티브 필터
- 채팅 인터페이스

### FastAPI
- 모던한 반응형 디자인
- REST API 엔드포인트
- 커스텀 HTML/CSS/JS
- 프로덕션 레벨

## 📦 배포

### Docker (추천)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit
CMD ["streamlit", "run", "web/streamlit_app.py", "--server.port", "8501"]

# 또는 FastAPI
# CMD ["uvicorn", "web.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t startup-matcher .
docker run -p 8501:8501 --env-file .env startup-matcher
```

### 클라우드 배포

**Streamlit Cloud (무료)**
1. GitHub에 푸시
2. streamlit.io/cloud 접속
3. 레포지토리 연결
4. 자동 배포

**AWS/GCP/Azure**
- Docker 컨테이너로 배포
- 또는 PaaS (Heroku, Railway 등)

## 🔍 API 사용 예시

### 매칭 요청

```bash
curl -X POST "http://localhost:8000/api/match" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "김창업",
    "age": 29,
    "region": "서울",
    "business_stage": "예비창업자",
    "business_field": "AI",
    "target_type": "청년"
  }'
```

### 챗봇 질문

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {...},
    "question": "1위 사업 신청 자격이 어떻게 되나요?",
    "history": []
  }'
```

## 🧪 테스트

```bash
# 전체 테스트
pytest

# 특정 모듈
pytest tests/test_agents.py

# 커버리지
pytest --cov=agents --cov-report=html
```

## 📈 성능 최적화

1. **캐싱**: 첫 실행 후 데이터 재사용
2. **임베딩**: Sentence-BERT로 빠른 검색
3. **비동기**: FastAPI 비동기 처리
4. **배치**: 여러 프로필 한 번에 처리

## 🛠️ 커스터마이징

### 새 데이터 소스 추가

```python
# agents/data_collector.py에 엔드포인트 추가
class APIEndpoint(Enum):
    NEW_SOURCE = "/path/to/new/api"

# collect_all() 메서드에 추가
data['new_source'] = self._fetch_endpoint(APIEndpoint.NEW_SOURCE, max_pages)
```

### 매칭 로직 수정

```python
# agents/semantic_matcher.py
def match(self, profile: UserProfile, top_k: int = 20):
    # 커스텀 로직 추가
    ...
```

## 🐛 트러블슈팅

### API 키 오류
```
ValueError: KISED_SERVICE_KEY 환경변수가 설정되지 않았습니다
```
→ `.env` 파일에 `KISED_SERVICE_KEY` 추가

### 임베딩 오류
```
ModuleNotFoundError: No module named 'sentence_transformers'
```
→ `pip install sentence-transformers`

### 포트 충돌
```
Error: Address already in use
```
→ 다른 포트 사용: `streamlit run --server.port 8502`

## 📝 라이선스

MIT License

## 👥 기여

Pull Request 환영합니다!

## 📧 문의

- 이슈: GitHub Issues
- 이메일: your@email.com

---

Made with ❤️ by AI Agents
