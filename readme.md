# Startup Support Retrieval System

FastAPI + React + `agentic_hybrid` 기반 창업지원 정보 검색/추천 시스템입니다.

## 구성

- 현재 서비스 엔진: [agentic_hybrid](agentic_hybrid)
- 서버 엔트리포인트: [web/fastapi_app.py](web/fastapi_app.py)
- 프런트엔드: [web/frontend](web/frontend)
- 현재 서비스가 읽는 인덱스: [cache](cache)
- 재수집/인덱스 재생성: [legacy_core](legacy_core), [legacy_agents](legacy_agents)

핵심 흐름:

1. 일반 질의/추천
   - Browser -> FastAPI -> `agentic_hybrid` -> `cache/rag_index.*`
2. 재수집/재생성
   - FastAPI -> `legacy_core.orchestrator.ensure_index(use_cache=False)`
   - `legacy_agents.data_collector.collect_all()`
   - `legacy_agents.rag_builder`
   - `cache/rag_index.*` 재생성
   - FastAPI 런타임 재초기화

## 요구사항

- Python 3.11 권장
- 의존성 설치:

```powershell
python -m pip install -r requirements.txt
```

## 환경변수

필수:

- `KISED_SERVICE_KEY`
  - 공공 원천 API 키
  - 재수집 시 없거나 잘못되면 `401 Unauthorized`

권장:

- `GOOGLE_API_KEY`
- `EMBEDDING_MODEL=BAAI/bge-m3`
- `AH_EMBEDDING_MODEL=BAAI/bge-m3`
- `AH_VECTORSTORE_DIR=cache`
- `AH_LLM_PROVIDER=google`
- `AH_LLM_MODEL=gemini-2.5-pro`

예시 `.env`:

```env
KISED_SERVICE_KEY=your_full_decoded_service_key
GOOGLE_API_KEY=your_google_api_key

EMBEDDING_MODEL=BAAI/bge-m3
AH_EMBEDDING_MODEL=BAAI/bge-m3
AH_VECTORSTORE_DIR=cache
AH_LLM_PROVIDER=google
AH_LLM_MODEL=gemini-2.5-pro
```

주의:

- `KISED_SERVICE_KEY`는 전체 키여야 합니다.
- `.env` 수정 후에는 서버를 완전히 재시작해야 합니다.

## 실행

`.env`에 키가 있으면 아래만 실행하면 됩니다.

```powershell
$env:EMBEDDING_MODEL="BAAI/bge-m3"
$env:AH_EMBEDDING_MODEL="BAAI/bge-m3"
$env:AH_VECTORSTORE_DIR="cache"
$env:AH_LLM_PROVIDER="google"
$env:AH_LLM_MODEL="gemini-2.5-pro"
$env:AH_GOOGLE_API_KEY=$env:GOOGLE_API_KEY

python -m uvicorn web.fastapi_app:app --host 0.0.0.0 --port 8001
```



## 현재 파이프라인

현재 웹앱은 `agentic_hybrid`를 사용합니다.

주요 노드:

- `query_expansion`
- `retrieve`
- `doc_type_router`
- `rerank`
- `planner`
- `inherit_deadline`
- `llm_deadline_review`
- `filter`
- `freshness_rerank`
- `dedup`
- `cross_doc_enrich`
- `final_policy_gate`
- `generate`
- `revise`

## 캐시와 재수집

- `캐시 사용 ON`
  - 현재 `cache` 인덱스 사용
- `캐시 사용 OFF`
  - 재수집 + 인덱스 재생성 후 새 cache로 추천

관리용 엔드포인트:

- `GET /api/admin/rebuild-index`
- `POST /api/admin/rebuild-index`

예시:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8001/api/admin/rebuild-index
Invoke-RestMethod http://127.0.0.1:8001/api/admin/rebuild-index
```

## 최신성 기준

`캐시 사용 OFF`를 선택하면 원천 API를 다시 호출하고, 기본적으로 **최신 90일 + 마감 제외** 기준으로 cache를 다시 만듭니다.

현재 기준:

- 엔드포인트별 최대 `5페이지`
- 페이지당 `100건`
- `ANNOUNCEMENT`
  - 마감 제외
  - 최근 `90일` 범위의 진행중 공고만 유지
- `BUSINESS`
  - 현재 연도 사업만 유지
- `CONTENT`, `STATISTICAL`
  - 최근 `90일` 데이터 우선 유지
- `CERT_PRODUCT`, `CERT_CORPORATE`
  - 유효기간 기준 유지

공고는 인덱싱 단계에서도 한 번 더 마감 여부와 90일 기준을 확인합니다.


## 레거시 경로

`legacy_core`, `legacy_agents`는 현재 질의 응답 엔진은 아니지만 아래에는 여전히 필요합니다.

- 데이터 재수집
- 인덱스 재생성
- CLI/실험/평가 

즉:

- `agentic_hybrid` = 현재 서비스 엔진
- `legacy_*` = 데이터 생산 및 구형 파이프라인
