# 🚀 Agentic AI 시스템 설치 및 실행 가이드

## 📝 목차

1. [사전 준비](#1-사전-준비)
2. [설치 과정](#2-설치-과정)
3. [API 키 발급](#3-api-키-발급)
4. [첫 실행](#4-첫-실행)
5. [문제 해결](#5-문제-해결)

---

## 1. 사전 준비

### 필수 소프트웨어

```bash
# Python 버전 확인
python --version  # 3.8 이상 필요

# pip 업그레이드
pip install --upgrade pip
```

### API 키 필요

1. **K-Startup API 키** (필수)
   - https://www.data.go.kr
   
2. **OpenAI API 키** (필수)
   - https://platform.openai.com/api-keys

---

## 2. 설치 과정

### Step 1: 프로젝트 다운로드

```bash
# 방법 1: Git Clone (권장)
git clone <repository-url>
cd startup-support-agentic

# 방법 2: ZIP 다운로드
# GitHub에서 "Code" → "Download ZIP" → 압축 해제
```

### Step 2: 가상환경 생성

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

가상환경이 활성화되면 프롬프트 앞에 `(venv)` 표시됨

### Step 3: 패키지 설치

```bash
pip install -r requirements.txt
```

**예상 시간**: 3-5분

### Step 4: 디렉토리 생성

```bash
# Windows
mkdir cache logs reports

# Mac/Linux
mkdir -p cache logs reports
```

---

## 3. API 키 발급

### 3-1. K-Startup API 키

1. https://www.data.go.kr 접속
2. 회원가입 / 로그인
3. "데이터 찾기" → "중소벤처기업진흥공단" 검색
4. 활용신청 클릭
5. 발급받은 키 복사

### 3-2. OpenAI API 키

1. https://platform.openai.com 접속
2. 로그인 / 회원가입
3. "API keys" 메뉴
4. "Create new secret key" 클릭
5. 키 복사 (다시 볼 수 없으니 안전하게 보관!)

### 3-3. .env 파일 생성

```bash
# .env.example을 복사
cp .env.example .env

# Windows
copy .env.example .env
```

`.env` 파일 편집:

```env
# 필수 - 발급받은 키를 붙여넣기
KISED_SERVICE_KEY=your_actual_kised_key_here
OPENAI_API_KEY=sk-your_actual_openai_key_here

# 선택 - 기본값 사용 가능
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
DATA_DAYS_RANGE=180
CACHE_REFRESH_DAYS=1
AGENT_VERBOSE=false
```

**⚠️ 주의**: 
- API 키는 절대 공개하지 마세요
- `.env` 파일은 Git에 커밋하지 마세요

---

## 4. 첫 실행

### 4-1. 설정 확인

```bash
python main.py recommend
```

**예상 출력**:
```
🤖 Agentic AI 시스템 초기화 중...
✅ LLM 클라이언트 준비 완료
✅ RAG 시스템 준비 완료
✅ 5개 도구 등록 완료
...
```

**오류가 나면**: [5. 문제 해결](#5-문제-해결) 참고

### 4-2. 웹 인터페이스 실행

```bash
streamlit run web/streamlit_app.py
```

브라우저가 자동으로 열리면서 `http://localhost:8501` 접속

**수동 접속**: 브라우저에서 위 주소 직접 입력

### 4-3. 첫 추천 받기

1. 왼쪽 사이드바에서 프로필 입력
   - 나이: 29
   - 지역: 서울
   - 창업단계: 예비창업자
   - 사업분야: AI
   
2. "✅ 프로필 저장" 클릭

3. "📊 맞춤추천" 탭으로 이동

4. "🎯 추천 받기" 클릭

5. 약 30초 후 결과 확인

### 4-4. 챗봇 사용하기

1. "💬 상담챗봇" 탭으로 이동

2. 빠른 질문 버튼 클릭하거나

3. 직접 질문 입력:
   - "서울 청년 지원 프로그램 알려줘"
   - "AI 창업 교육 뭐 있어?"
   
4. "전송 📤" 클릭

---

## 5. 문제 해결

### 문제 1: API 키 오류

```
❌ ValueError: KISED_SERVICE_KEY 환경변수가 설정되지 않았습니다
```

**해결**:
1. `.env` 파일이 있는지 확인
2. 파일 내용 확인 (키가 올바른지)
3. 따옴표 없이 입력했는지 확인

```env
# ❌ 잘못된 예
KISED_SERVICE_KEY="your_key"

# ✅ 올바른 예
KISED_SERVICE_KEY=your_key
```

### 문제 2: 패키지 설치 오류

```
ERROR: Could not find a version that satisfies the requirement
```

**해결**:
```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 설치 시도
pip install streamlit
pip install sentence-transformers
pip install openai
```

### 문제 3: 메모리 부족

```
MemoryError: Unable to allocate array
```

**해결**:
`.env` 파일 수정:
```env
MAX_PAGES_PER_ENDPOINT=2  # 3에서 2로 줄이기
```

### 문제 4: 포트 충돌

```
Port 8501 is already in use
```

**해결**:
```bash
# 다른 포트 사용
streamlit run web/streamlit_app.py --server.port 8502
```

### 문제 5: Sentence Transformers 설치 오류

```bash
# CPU 버전 설치
pip install sentence-transformers --no-deps
pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 문제 6: 인코딩 오류 (Windows)

```
UnicodeDecodeError: 'cp949' codec can't decode
```

**해결**:
```bash
# PowerShell에서 실행
$env:PYTHONIOENCODING="utf-8"
streamlit run web/streamlit_app.py
```

### 문제 7: 데이터 수집 실패

```
⚠ HTTP 500: Internal Server Error
```

**해결**:
1. API 키가 유효한지 확인
2. 활용신청 승인 확인
3. 잠시 후 재시도

---

## 6. 성능 최적화 팁

### 첫 실행이 느릴 때

첫 실행 시 다음 작업이 진행됩니다:
- Sentence Transformers 모델 다운로드 (~500MB)
- API 데이터 수집
- RAG 인덱스 구축

**예상 시간**: 3-10분

이후 실행은 캐시를 사용하여 빠릅니다 (5-10초).

### 캐시 수동 삭제

```bash
# Windows
rmdir /s cache

# Mac/Linux
rm -rf cache
```

다음 실행 시 자동으로 재생성됩니다.

---

## 7. 다음 단계

### CLI 테스트

```bash
# 전체 테스트
python main.py all

# 대화형 모드
python main.py interactive
```

### 코드 탐색

1. `core/tools.py` - 도구 시스템
2. `agents/agentic_base.py` - 에이전트 로직
3. `web/streamlit_app.py` - UI 커스터마이징

### 설정 조정

`.env` 파일에서:
- `AGENT_VERBOSE=true` - 상세 로그 보기
- `DATA_DAYS_RANGE=90` - 더 빠른 수집
- `LLM_MODEL=gpt-4` - 더 나은 품질 (비용↑)

---

## 8. 추가 자원

- 📖 [README.md](README.md) - 전체 문서
- 🐛 [GitHub Issues](issues-url) - 버그 리포트
- 💬 [Discussions](discussions-url) - 질문/토론

---

**설치 완료를 축하합니다! 🎉**

문제가 계속되면 이슈를 등록해주세요.