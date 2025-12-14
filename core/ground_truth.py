# core/ground_truth.py
from typing import List, Dict, Any, Optional
from models.data import UserProfile


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).strip()


def _match_region(meta_region: str, profile_region: str) -> bool:
    """
    지역 매칭 규칙
    - meta_region이 '전국'이면 모두 허용
    - 비어있으면 허용
    - 그 외에는 profile_region이 포함되어 있으면 매칭
    """
    mr = _norm(meta_region)
    pr = _norm(profile_region)

    if not pr:
        return True  # 프로필에 지역 정보 없으면 패스

    if not mr:
        return True  # 메타에 지역이 없으면 너무 빡세게 막지 않음

    if mr == "전국":
        return True

    return pr in mr  # 예: mr = "부산광역시", pr = "부산"


def _match_stage(meta_stage: str, profile_stage: str) -> bool:
    """
    단계(예비창업, 초기창업 등) 매칭
    - 메타가 비어있으면 허용
    - profile_stage 문자열이 meta_stage에 포함되어 있으면 매칭
    """
    ms = _norm(meta_stage)
    ps = _norm(profile_stage)

    if not ps:
        return True
    if not ms:
        return True

    return ps in ms


def _match_field(meta_field: str, profile_field: str) -> bool:
    """
    업종/분야 매칭
    - 둘 다 비어있으면 허용
    - profile_field가 meta_field 안에 포함되면 매칭
    """
    mf = _norm(meta_field)
    pf = _norm(profile_field)

    if not pf:
        return True
    if not mf:
        return True

    return pf in mf


def _match_target_type(meta_target: str, target_type: str) -> bool:
    """
    지원 대상(청년, 여성, 장애인 등) 매칭
    - target_type이 비어있으면 모두 허용
    - meta_target 안에 '청년', '여성', '예비창업자' 등 키워드가 있으면 매칭
    """
    mt = _norm(meta_target)
    tt = _norm(target_type)

    if not tt:
        return True
    if not mt:
        return True

    # 예: 프로필 target_type에 "청년"이 들어 있으면,
    # meta_target 안에 "청년"이 있으면 매칭
    return tt in mt


def _match_status(meta_status: str) -> bool:
    """
    마감 여부 필터
    - '마감', '종료' 같은 상태는 ground truth에서 제외
    """
    ms = _norm(meta_status)
    bad_keywords = ["마감", "종료", "접수종료"]

    for k in bad_keywords:
        if k in ms:
            return False
    return True


def build_rule_based_ground_truth(
    profile: UserProfile,
    rag_system: object,
    base_query: str = "창업 지원사업",
    top_k: int = 300,
    desired_data_types: Optional[List[str]] = None,
) -> List[str]:
    """
    기존 인덱스 + 메타데이터 + 규칙으로 pseudo ground truth 만드는 함수

    - profile: UserProfile (region, business_stage, business_field, target_type 사용)
    - rag_system: RAGSystem 인스턴스 (rag_system.search 사용)
    - base_query: 넓게 검색할 기본 쿼리
    - top_k: ground truth 후보로 볼 문서 개수
    - desired_data_types: ["announcement"] 같이 data_type 필터를 걸고 싶을 때

    반환:
        ground_truth_ids: ["DOC_001", "DOC_007", ...]
    """
    filters: Dict[str, Any] = {}
    if desired_data_types:
        # RAGBackend에서 $in 필터 지원하도록 되어 있음
        filters["data_type"] = {"$in": desired_data_types}

    # 1) 넓게 검색
    results = rag_system.search(
        query=base_query,
        top_k=top_k,
        filters=filters or None,
    )

    gt_ids: List[str] = []

    for r in results:
        doc = r.get("document")
        if doc is None:
            continue

        meta = getattr(doc, "metadata", {}) or {}

        meta_region = meta.get("region", "")
        meta_stage = meta.get("business_stage", "") or meta.get("stage", "")
        meta_field = meta.get("field", "") or meta.get("biz_type", "")
        meta_target = meta.get("apply_target", "") or meta.get("target", "")
        meta_status = meta.get("status", "")

        # 2) 규칙 기반 필터 적용
        if not _match_region(meta_region, getattr(profile, "region", "")):
            continue
        if not _match_stage(meta_stage, getattr(profile, "business_stage", "")):
            continue
        if not _match_field(meta_field, getattr(profile, "business_field", "")):
            continue
        if not _match_target_type(meta_target, getattr(profile, "target_type", "")):
            continue
        if not _match_status(meta_status):
            continue

        # 3) 최종 id 추출
        doc_id = getattr(doc, "id", None) or meta.get("id")
        if not doc_id:
            continue

        gt_ids.append(str(doc_id))

    # 중복 제거
    gt_ids = list(dict.fromkeys(gt_ids))

    return gt_ids
