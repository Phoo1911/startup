"""
rag_builder_patch.py — RAGBuilderAgent에 추가할 _validate_documents() 메서드

기존 rag_builder_agent.py의 RAGBuilderAgent 클래스에 아래 메서드들을 추가한다.

추가 위치: build_index() 메서드 바로 아래

사용 예:
    count = builder.build_index(raw_data)
    report = builder.validate_index()
    print(report)
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional


# ── 아래 메서드들을 RAGBuilderAgent 클래스에 추가 ──────────────────────────


def _validate_documents(self, documents: List[Any]) -> Dict[str, Any]:
    """
    인덱스에 추가된 문서들의 품질을 점검하고 리포트를 반환한다.

    Returns
    -------
    dict with keys:
      total          : 전체 문서 수
      type_counts    : 타입별 카운트
      empty_text     : 텍스트가 너무 짧은 문서 수 (< 20자)
      missing_title  : 타이틀 없는 문서 수
      missing_url    : detail_url 없는 문서 수
      expired        : 이미 만료된 문서 수 (deadline < today)
      missing_deadline_ann : ANN 타입인데 deadline 없는 수
      no_age_limit_ann     : ANN 타입인데 age_limit 없는 수
      warnings       : 문자열 경고 목록
    """
    today = datetime.now().date()
    type_counts: Dict[str, int] = {}
    empty_text = 0
    missing_title = 0
    missing_url = 0
    expired = 0
    missing_deadline_ann = 0
    no_age_limit_ann = 0
    warnings: List[str] = []

    for doc in documents:
        meta = getattr(doc, "metadata", None) or {}
        if isinstance(doc, dict):
            meta = doc.get("metadata", {}) or {}

        doc_type = str(meta.get("type") or "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        # 텍스트 길이
        text = str(getattr(doc, "text", None) or doc.get("text", "") if isinstance(doc, dict) else "")
        if len(text.strip()) < 20:
            empty_text += 1

        # 타이틀
        title = str(meta.get("title") or "")
        if not title.strip():
            missing_title += 1

        # URL
        url = str(meta.get("detail_url") or meta.get("guide_url") or meta.get("apply_url") or "")
        if not url.strip():
            missing_url += 1

        # 마감일 확인
        deadline_raw = str(meta.get("deadline") or "")
        if deadline_raw:
            d = self._parse_date_for_validation(deadline_raw)
            if d and d < today:
                expired += 1

        # ANN 전용 체크
        if doc_type == "announcement":
            if not deadline_raw:
                missing_deadline_ann += 1
            age_limit = str(meta.get("age_limit") or "")
            if not age_limit:
                no_age_limit_ann += 1

    # 경고 생성
    total = len(documents)
    if total == 0:
        warnings.append("⚠️  문서가 0개입니다. raw_data를 확인하세요.")
    if empty_text / max(total, 1) > 0.05:
        warnings.append(f"⚠️  텍스트 짧은 문서 비율 {empty_text/total:.0%} > 5%")
    if expired / max(total, 1) > 0.3:
        warnings.append(f"⚠️  만료 문서 비율 {expired/total:.0%} > 30% — 데이터 갱신 필요")
    if "announcement" in type_counts and missing_deadline_ann / max(type_counts["announcement"], 1) > 0.2:
        warnings.append(f"⚠️  ANN 문서 중 deadline 없는 비율 높음: {missing_deadline_ann}/{type_counts['announcement']}")

    return {
        "total": total,
        "type_counts": type_counts,
        "empty_text": empty_text,
        "missing_title": missing_title,
        "missing_url": missing_url,
        "expired": expired,
        "missing_deadline_ann": missing_deadline_ann,
        "no_age_limit_ann": no_age_limit_ann,
        "warnings": warnings,
    }


def _parse_date_for_validation(self, value: str) -> Optional[date]:
    """검증용 날짜 파싱 (순수 stdlib)."""
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 8:
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        except ValueError:
            pass
    return None


def validate_index(self) -> str:
    """
    RAG 시스템에서 문서를 가져와 _validate_documents()를 실행하고
    사람이 읽기 좋은 리포트 문자열을 반환한다.

    사용법:
        builder = RAGBuilderAgent(rag_system)
        builder.build_index(raw_data)
        print(builder.validate_index())
    """
    # rag 시스템에서 문서 목록 획득 (인터페이스에 따라 조정 필요)
    docs = []
    if hasattr(self.rag, "documents"):
        docs = self.rag.documents
    elif hasattr(self.rag, "get_all_documents"):
        docs = self.rag.get_all_documents()

    report = self._validate_documents(docs)

    lines = [
        "=" * 50,
        "  RAG Index Validation Report",
        "=" * 50,
        f"  총 문서 수       : {report['total']:,}",
        "",
        "  [타입별 분포]",
    ]
    for t, cnt in sorted(report["type_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"    {t:<20}: {cnt:>5,}")

    lines += [
        "",
        "  [품질 지표]",
        f"    텍스트 짧은 문서  : {report['empty_text']}",
        f"    타이틀 없는 문서  : {report['missing_title']}",
        f"    URL 없는 문서    : {report['missing_url']}",
        f"    이미 만료된 문서  : {report['expired']}",
        f"    ANN deadline 없음: {report['missing_deadline_ann']}",
        f"    ANN age_limit 없음: {report['no_age_limit_ann']}",
    ]

    if report["warnings"]:
        lines += ["", "  [경고]"]
        for w in report["warnings"]:
            lines.append(f"    {w}")
    else:
        lines.append("\n  ✅ 모든 품질 기준 통과")

    lines.append("=" * 50)
    return "\n".join(lines)
