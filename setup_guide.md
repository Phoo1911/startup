# 🚀 설치 및 실행 가이드

## 📋 목차
1. [사전 준비](#사전-준비)
2. [프로젝트 구조 생성](#프로젝트-구조-생성)
3. [설치 방법](#설치-방법)
4. [실행 방법](#실행-방법)
5. [문제 해결](#문제-해결)

---

## 사전 준비

### 필수 요구사항
- **Python 3.9 이상**
- **K-Startup API 키** ([신청 링크](https://www.data.go.kr))

### 선택사항
- OpenAI API 키 (LLM 기능 사용 시)
- Docker (컨테이너 실행 시)

---

## 프로젝트 구조 생성

### 1. 폴더 구조 만들기

```bash
mkdir startup-matching-system
cd startup-matching-system

# 하위 폴더 생성
mkdir -p config models agents core utils web/templates web/static cache reports tests
```

### 2. 파일 배치

아티팩트에서 제공된 코드를 다음과 같이 배치:

```
startup-matching-system/
├── config/
│   ├── __init__.py
│   └── settings.py
├── models/
│   ├── __init__.py
│   ├── data_models.py
│   └── enums.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── data_collector.py
│   ├── rag_builder.py
│   ├── semantic_matcher.py
│   ├── llm_reasoner.py
│   ├── recommender.py
│   └── chatbot.py
├── core/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── rag_system.py
│   └── orchestrator.py
├── utils/
│   ├── __init__.py
│   ├── text_utils.py
│   └── date_utils.py
├── web/
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── fastapi_app.py
│   └── templates/
│       └── index.html
├── .env
├── .gitignore
├── requirements.txt
├── main.py
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 설치 방법

### Option 1: 로컬 설치 (권장)

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt
```

### Option 2: Docker 설치

```bash
# Docker 이미지 빌드
docker-compose build

# 실행 (Streamlit)
docker-compose up streamlit

# 실행 (FastAPI)
docker-compose --profile api up fastapi
```

---

## 환경변수 설정

### 1. .env 파일 생성

```bash
cp .env.example .env
```

### 2. .env 파일 편집

```bash
# 필수
KISED_SERVICE_KEY=your_api_key_here

# LLM 사용 시 (선택)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# 또는 LLM 비활성화
LLM_PROVIDER=
```

---

## 실행 방법

### 🎨 Streamlit 웹 UI (가장 쉬움)

```bash
streamlit run web/streamlit_app.py
```

**접속**: http://localhost:8501

**특징**:
- ✅ 가장 빠른 시작
- ✅ 인터랙티브 UI
- ✅ 실시간 결과 확인
- ❌ 커스터마이징 제한

---

### ⚙️ FastAPI 웹 서버 (프로덕션)

```bash
uvicorn web.fastapi_app:app --reload
```

**접속**: 
- 웹: http://localhost:8000
- API 문서: http://localhost:8000/docs

**특징**:
- ✅ REST API 제공
- ✅ 완전한 커스터마이징
- ✅ 프로덕션 레벨
- ❌ 프론트엔드 직접 구현 필요

---

### 💻 CLI 실행

```bash
# 기본 실행
python main.py --name "김창업" --region "서울" --field "AI"

# 자세한 옵션
python main.py \
  --name "이스타트" \
  --age 35 \
  --region "부산" \
  --stage "3년이하" \
  --field "제조" \
  --target "여성" \
  --context "친환경 제품 개발" \
  --top-n 10 \
  --verbose
```

---

## 주요 기능 테스트

### 1. 데이터 수집 테스트

```python
from core.orchestrator import AgenticOrchestrator
from models.data_models import UserProfile

orchestrator = AgenticOrchestrator(
    service_key="YOUR_KEY",
    llm_api_key="YOUR_LLM_KEY"  # 선택사항
)

profile = UserProfile(
    name="테스트",
    age=29,
    region="서울",
    business_stage="예비창업자",
    business_field="AI",
    target_type="청년"
)

report = orchestrator.run(profile, top_n=5, use_cache=False)
print(f"총 {report['total_matches']}개 매칭")
```

### 2. 챗봇 테스트

```python
question = "AI 관련 지원사업 중에서 마감이 임박한 게 있나요?"
answer = orchestrator.chatbot.chat(profile, question)
print(answer)
```

---

## 문제 해결

### ❌ ModuleNotFoundError

```bash
# 패키지 재설치
pip install -r requirements.txt --upgrade
```

### ❌ API 키 오류

```
ValueError: KISED_SERVICE_KEY 환경변수가 설정되지 않았습니다
```

**해결**: `.env` 파일에 API 키 추가

### ❌ 포트 충돌

```
Error: Address already in use
```

**해결**: 다른 포트 사용
```bash
streamlit run web/streamlit_app.py --server.port 8502
uvicorn web.fastapi_app:app --port 8001
```

### ❌ 임베딩 모델 오류

```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**해결**:
```bash
pip install sentence-transformers
```

또는 `.env`에서:
```bash
EMBEDDING_MODEL=simple
```

### ❌ LLM 오류

```
[LLM 오류: ...]
```

**해결**:
1. API 키 확인
2. LLM 비활성화하고 RAG만 사용:
```bash
LLM_PROVIDER=
```

---

## 성능 최적화

### 1. 캐시 활성화

```python
# 첫 실행 (느림)
report = orchestrator.run(profile, use_cache=False)

# 이후 실행 (빠름)
report = orchestrator.run(profile, use_cache=True)
```

### 2. 자동 새로고침

```python
from core.orchestrator import run_with_auto_refresh

# 하루에 한 번만 새로 수집
report = run_with_auto_refresh(
    orchestrator, 
    profile, 
    top_n=10,
    refresh_days=1  # 1일마다 갱신
)
```

### 3. 임베딩 모델 선택

```bash
# 빠르지만 정확도 낮음
EMBEDDING_MODEL=simple

# 느리지만 정확도 높음
EMBEDDING_MODEL=sentence-transformers
```

---

## 배포

### Streamlit Cloud (무료, 추천)

1. GitHub에 푸시
2. https://streamlit.io/cloud 접속
3. 레포지토리 연결
4. Secrets에 환경변수 추가
5. 자동 배포

### Docker

```bash
# 빌드
docker build -t startup-matcher .

# 실행
docker run -p 8501:8501 --env-file .env startup-matcher
```

### AWS/GCP/Azure

```bash
# 예: AWS EC2
ssh your-server
git clone your-repo
cd startup-matching-system
pip install -r requirements.txt
streamlit run web/streamlit_app.py --server.port 80
```

---

## 다음 단계

1. ✅ 시스템 설치 및 실행
2. 📊 샘플 프로필로 테스트
3. 🎨 웹 UI 커스터마이징
4. 🔧 추가 데이터 소스 연동
5. 🚀 프로덕션 배포

---

궁금한 점이 있으면 Issues에 올려주세요! 🙋‍♂️
