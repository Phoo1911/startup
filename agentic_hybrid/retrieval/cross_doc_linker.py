"""
retrieval/cross_doc_linker.py — ANN ↔ BIZ 문서 간 연결 및 메타데이터 상속

문제:
  - BIZ 문서는 deadline이 항상 "" (빈 문자열)
  - ANN 문서가 intg_pbanc_biz_nm으로 BIZ 문서를 참조하지만 현재 연결이 없음
  - 동일 사업의 BIZ와 ANN이 각각 검색 결과에 독립적으로 나타남

해결:
  1. build_index() — intg_pbanc_biz_nm → ANN 문서 역인덱스 구축
  2. enrich_biz_deadline() — BIZ 문서의 빈 deadline을 연결된 ANN에서 상속
  3. enrich_retrieved() — 검색 결과에 연결 문서 정보를 side_data로 병합

사용 예:
    linker = CrossDocLinker(all_documents)
    enriched_docs = linker.enrich_retrieved(retrieved_docs)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


class CrossDocLinker:
    """
    ANN(공고)과 BIZ(지원사업) 문서 간의 양방향 연결 인덱스를 구축하고
    메타데이터를 상호 보완한다.
    """

    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        # title → BIZ doc 매핑
        self._biz_by_title: Dict[str, Dict[str, Any]] = {}
        # intg_pbanc_biz_nm → ANN doc 리스트
        self._ann_by_biz_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # doc id → linked doc ids
        self._links: Dict[str, List[str]] = defaultdict(list)

        self._build(documents)

    # ── 인덱스 구축 ──────────────────────────────────────────────────────

    def _build(self, documents: List[Dict[str, Any]]) -> None:
        # 1차 패스: BIZ 문서 타이틀 인덱싱
        for doc in documents:
            meta = doc.get("metadata", {}) or {}
            if meta.get("type") == "business":
                title = str(meta.get("title") or doc.get("title") or "").strip()
                if title:
                    self._biz_by_title[title] = doc

        # 2차 패스: ANN 문서의 intg_pbanc_biz_nm으로 BIZ 연결
        for doc in documents:
            meta = doc.get("metadata", {}) or {}
            if meta.get("type") != "announcement":
                continue
            biz_name = str(meta.get("intg_biz_name") or "").strip()
            if not biz_name:
                continue

            self._ann_by_biz_name[biz_name].append(doc)

            # 양방향 링크 기록
            ann_id = doc.get("id", "")
            biz_doc = self._biz_by_title.get(biz_name)
            if biz_doc:
                biz_id = biz_doc.get("id", "")
                if ann_id and biz_id:
                    self._links[ann_id].append(biz_id)
                    self._links[biz_id].append(ann_id)

    # ── BIZ deadline 상속 ────────────────────────────────────────────────

    def enrich_biz_deadline(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        BIZ 문서의 deadline=""인 경우, 연결된 ANN 문서 중 가장 늦은
        deadline을 상속한다. 원본 리스트를 직접 수정하지 않고 복사본 반환.
        """
        enriched = []
        for doc in documents:
            doc = dict(doc)
            meta = dict(doc.get("metadata", {}) or {})
            doc["metadata"] = meta

            if meta.get("type") != "business":
                enriched.append(doc)
                continue

            # deadline이 이미 있으면 건드리지 않음
            if meta.get("deadline"):
                enriched.append(doc)
                continue

            title = str(meta.get("title") or doc.get("title") or "").strip()
            linked_anns = self._ann_by_biz_name.get(title, [])

            # 가장 늦은 deadline 선택
            latest = self._latest_deadline(linked_anns)
            if latest:
                meta["deadline"] = latest
                meta["deadline_source"] = "inherited_from_ann"

            enriched.append(doc)
        return enriched

    @staticmethod
    def _latest_deadline(anns: List[Dict[str, Any]]) -> Optional[str]:
        """ANN 목록에서 가장 늦은 deadline 문자열을 반환."""
        deadlines = []
        for ann in anns:
            d = str((ann.get("metadata") or {}).get("deadline") or "").strip()
            if d and len(d) >= 8:
                deadlines.append(d)
        return max(deadlines) if deadlines else None

    # ── 검색 결과 보강 ───────────────────────────────────────────────────

    def enrich_retrieved(
        self,
        docs: List[Dict[str, Any]],
        max_linked: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        검색된 각 문서에 연결 문서 요약 정보를 ``linked_docs`` 키로 추가한다.

        ANN 문서 → 연결된 BIZ의 category, budget, content 요약 추가
        BIZ 문서 → 연결된 ANN의 deadline, status, apply_url 추가

        Parameters
        ----------
        docs        : 검색/재정렬된 문서 리스트
        max_linked  : 문서당 연결 정보 최대 개수
        """
        enriched = []
        for doc in docs:
            doc = dict(doc)
            meta = doc.get("metadata", {}) or {}
            doc_id = doc.get("id", "")
            doc_type = meta.get("type", "")

            linked_summaries: List[Dict[str, Any]] = []

            if doc_type == "announcement":
                # ANN → 연결 BIZ 정보
                biz_name = str(meta.get("intg_biz_name") or "").strip()
                biz_doc = self._biz_by_title.get(biz_name)
                if biz_doc:
                    biz_meta = biz_doc.get("metadata", {}) or {}
                    linked_summaries.append({
                        "linked_type": "business",
                        "linked_id": biz_doc.get("id"),
                        "category": biz_meta.get("category_cd"),
                        "budget": biz_meta.get("budget"),
                        "content_summary": str(biz_meta.get("content") or "")[:200],
                    })

            elif doc_type == "business":
                # BIZ → 연결 ANN 정보 (최신 공고 순)
                title = str(meta.get("title") or doc.get("title") or "").strip()
                linked_anns = self._ann_by_biz_name.get(title, [])
                # deadline 내림차순 정렬
                linked_anns = sorted(
                    linked_anns,
                    key=lambda a: str((a.get("metadata") or {}).get("deadline") or ""),
                    reverse=True,
                )
                for ann in linked_anns[:max_linked]:
                    ann_meta = ann.get("metadata", {}) or {}
                    linked_summaries.append({
                        "linked_type": "announcement",
                        "linked_id": ann.get("id"),
                        "deadline": ann_meta.get("deadline"),
                        "status": ann_meta.get("status"),
                        "apply_url": ann_meta.get("apply_url"),
                        "title": ann_meta.get("title"),
                    })

            if linked_summaries:
                doc["linked_docs"] = linked_summaries

            enriched.append(doc)
        return enriched

    # ── 통계 ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "biz_indexed": len(self._biz_by_title),
            "ann_linked": sum(len(v) for v in self._ann_by_biz_name.values()),
            "total_links": sum(len(v) for v in self._links.values()) // 2,
        }
