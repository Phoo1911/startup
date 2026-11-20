# Agentic AI 기반 창업지원 질의응답·추천 시스템

창업지원 공고·사업·인증 정보를 공공 API에서 모아, 사용자의 지역·나이·업력 조건에 맞는 지원사업을 찾고 근거와 함께 답하는 시스템입니다. 석사학위논문의 구현 코드이며, 그 전 단계인 AutoRAG 학술지 연구의 데이터 수집·평가 코드도 함께 들어 있습니다.

| 연구 | 내용 | 결과 |
| --- | --- | --- |
| [석사학위논문](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=91bd0634486716beffe0bdc3ef48d419) 「에이전틱 AI 기반 창업지원 질의응답 및 추천 시스템」 (인제대학교, 2026) | 같은 검색 모델로 Standard RAG와 이 저장소의 Agentic 그래프 비교 | 547개 QA, LLM 평가 기준 답변 정확성 0.7528 → 0.7789, 근거 충실성 0.8462 → 0.8913 |
| [학술지 논문](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003338739) 「의미 기반 임베딩을 이용하는 AutoRAG 기반 공공창업지원 시스템 개발 및 성능 분석」 (한국정보통신학회논문지 30(5), 2026, 제1저자) | 공공 API 데이터 표준화, 임베딩 모델별 검색·생성 성능 비교 | 최고 모델 기준 Top-1 F1 0.810, Top-10 nDCG 0.779 |

수치는 각 논문의 데이터·평가 조건에서 나온 값입니다. 실제 사용자 대상 정확도나 운영 성과가 아닙니다.

## 왜 검색만으로는 부족했나

AutoRAG 연구에서 관련 문서를 잘 찾는 것까지는 확인했지만, 실제 질문에는 "만 39세 이하", "창업 3년 이내", "경남 소재" 같은 자격 조건이 붙습니다. 공고와 사업 정보가 서로 다른 API에 나뉘어 있어 마감일이 사업 문서에는 없고 공고 문서에만 있는 경우도 많았습니다. 그래서 검색 뒤에 조건을 확인하고, 연결된 문서의 정보를 합치고, 답변을 다시 검토하는 단계를 LangGraph로 붙였습니다.

## 처리 흐름

`agentic_hybrid/graph.py`

```
질문 + 사용자 프로필
  → intent_classifier      질문이 원하는 문서 유형 판단 (공고 / 사업 / 콘텐츠 / 통계 / 인증)
  → query_expansion        검색어 확장
  → retrieve               벡터 검색 + BM25
  → doc_type_router        문서 유형·지역 기준 재정렬 (규칙 기반)
  → rerank                 cross-encoder 재순위
  → planner                질문에서 나이·업력·지역·업종 조건 추출 (프로필 값이 우선)
  → inherit_deadline       사업 문서에 연결된 공고의 마감일 상속
  → llm_deadline_review    규칙으로 판정 못 한 마감 여부만 LLM이 보조 판정
  → filter                 나이·업력·지역·업종·마감 조건으로 제외
  → freshness_rerank       최신성 보정
  → dedup                  같은 사업의 중복 문서 정리
  → cross_doc_enrich       연결된 공고·사업 정보 병합
  → final_policy_gate      최종 조건 재확인
  → generate               근거 문서로 답변 생성
  → revise                 답변 검토, 근거가 부족하면 재검색 후 수정
```

비교용 `baseline_pure` 모드는 `retrieve → generate`만 실행합니다.

## 폴더 구성

| 경로 | 역할 |
| --- | --- |
| `agentic_hybrid/` | 현재 질의응답 엔진 (LangGraph 그래프, 노드, 검색기, 조건 판정 도구) |
| `web/fastapi_app.py`, `web/frontend/` | API 서버와 React 화면 |
| `legacy_agents/`, `legacy_core/` | 공공 API 수집과 인덱스 생성. 질의응답에는 쓰지 않지만 재수집에 필요 |
| `autorag_patch/`, `autorag_workspace/` | AutoRAG 실험 설정, QA 데이터, 평가 결과 표 |
| `run_agent_eval.py`, `evaluate_*.py`, `experiments/` | 모드별 답변 생성과 평가 |
| `weekly_rebuild.py` | 명령줄에서 재수집·인덱스 재생성 |

## 실행

```bash
pip install -r requirements.txt          # 웹 앱·질의응답·재수집
pip install -r requirements-eval.txt     # 평가 스크립트까지 돌릴 때
```

AutoRAG 청킹·QA 생성(`autorag_patch/`)은 의존성이 많아 별도 가상환경에서 `requirements-autorag.txt`로 설치합니다.

`.env`:

```env
KISED_SERVICE_KEY=공공데이터포털_인증키(디코딩된_전체_키)
GOOGLE_API_KEY=...

EMBEDDING_MODEL=BAAI/bge-m3
AH_EMBEDDING_MODEL=BAAI/bge-m3
AH_VECTORSTORE_DIR=cache
AH_LLM_PROVIDER=google
AH_LLM_MODEL=gemini-2.5-pro
AH_GOOGLE_API_KEY=...   # GOOGLE_API_KEY와 같은 값
```

서버 실행:

```bash
python -m uvicorn web.fastapi_app:app --host 0.0.0.0 --port 8001
```

**처음 클론한 경우 인덱스가 없습니다.** `cache/`는 저장소에 포함하지 않으므로 서버는 뜨지만 `/api/health`의 `agentic_hybrid`가 `false`입니다. 먼저 공공 API를 수집해 인덱스를 만듭니다 (`KISED_SERVICE_KEY` 필요).

```bash
curl -X POST http://127.0.0.1:8001/api/admin/rebuild-index   # 시작
curl http://127.0.0.1:8001/api/admin/rebuild-index           # 진행 상태
```

끝나면 런타임이 새 인덱스로 다시 초기화되고 http://127.0.0.1:8001 에서 사용할 수 있습니다. 화면의 "캐시 사용"을 끄고 추천을 요청해도 같은 재수집이 실행됩니다.

### 수집 기준

재수집할 때는 마감된 공고를 빼고 최근 데이터 위주로 인덱스를 만듭니다.

| 문서 유형 | 유지 기준 |
| --- | --- |
| 공고 (ANNOUNCEMENT) | 마감 제외, 최근 90일 내 진행 중 공고 |
| 사업 (BUSINESS) | 올해 사업 |
| 콘텐츠·통계 (CONTENT, STATISTICAL) | 최근 90일 우선 |
| 인증 (CERT_PRODUCT, CERT_CORPORATE) | 유효기간 기준 |

API 엔드포인트마다 최대 5페이지 × 100건을 가져오고, 인덱싱 단계에서 마감 여부를 한 번 더 확인합니다.

## 평가 재현

```bash
# 모드별 답변 생성 (예: 기본 RAG vs 전체 그래프)
python run_agent_eval.py --modes baseline_pure no_intent_dense --limit 20

# LLM 판정 기반 환각·근거 평가
python evaluate_hallucination.py

# 정책 조건 정확도
python evaluate_policy_accuracy.py
```

AutoRAG 검색·생성 평가 설정은 `autorag_workspace/configs/`에 있습니다. 

## 한계

- 평가는 연구용 QA 세트 기준입니다. 실제 사용자 질문과 운영 환경에서는 검증하지 않았습니다.
- 공공 API 응답 형식이나 필드가 바뀌면 수집 코드(`legacy_agents/data_collector.py`) 수정이 필요합니다.
- 공개 배포는 하지 않았습니다.
