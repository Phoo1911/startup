"""
tools/deadline_tool.py — Application deadline eligibility check

Fixes applied:
  - python-dateutil is an optional dependency; added a pure-stdlib fallback
    _parse_date_stdlib() so the system degrades gracefully instead of crashing
    with ImportError when dateutil is not installed.
  - Both parsers are tried in order: dateutil first (more robust for Korean
    date strings), stdlib fallback second.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, Optional


# ── Date parsing ──────────────────────────────────────────────────────────

_STDLIB_FORMATS = (
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y/%m/%d",
    "%Y%m%d",
)

_KOREAN_DATE_RE = re.compile(r"(\d{4})[^\d](\d{1,2})[^\d](\d{1,2})")
_DATE_TOKEN_RE = re.compile(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{8}")


def _parse_date_stdlib(value: str) -> Optional[date]:
    """Pure stdlib date parser — no third-party deps required."""
    value = value.strip()
    for fmt in _STDLIB_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    m = _KOREAN_DATE_RE.search(value)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    s = str(value).strip()

    # FIX: try dateutil first; fall back to stdlib so we don't crash if it's absent
    try:
        from dateutil import parser as du_parser  # type: ignore[import]
        return du_parser.parse(s).date()
    except ImportError:
        pass
    except Exception:
        pass

    return _parse_date_stdlib(s)


def _extract_period_end_date(metadata: Dict[str, Any]) -> Optional[date]:
    """
    Parse end date from application/support period text.

    Examples:
      - "2025-12-08 ~ 2026-01-16" -> 2026-01-16
      - "2025.12.08~2026.01.16"   -> 2026-01-16
      - "20251208~20260116"       -> 2026-01-16
    """
    period_fields = (
        "apply_period",
        "support_period",
        "recruit_period",
        "reception_period",
    )

    for key in period_fields:
        raw = metadata.get(key)
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue

        # Prefer the last date token in the period string as the end date.
        tokens = _DATE_TOKEN_RE.findall(text)
        if tokens:
            end_d = _parse_date(tokens[-1])
            if end_d is not None:
                return end_d

        # Fallback: split by common range separators and parse the tail.
        for sep in ("~", "∼", "-", "–", "—"):
            if sep in text:
                tail = text.rsplit(sep, 1)[-1].strip()
                end_d = _parse_date(tail)
                if end_d is not None:
                    return end_d
    return None


# ── Main check ────────────────────────────────────────────────────────────

_OPEN_STATUSES = {"Y", "모집중", "기본"}
_CLOSED_STATUSES = {"N", "마감", "종료", "접수마감", "마감완료", "모집마감"}


def passes_deadline_constraint(
    metadata: Dict[str, Any],
    today: Optional[date] = None,
) -> bool:
    """
    Returns True if the policy is still accepting applications.

    Checks (in order):
      1. ``status`` field — hard-closed statuses immediately return False.
      2. ``deadline`` or ``confmdoc_expr_dt`` date field — if set and in the
         past, returns False.
      3. If not available, parse the end date from a period field such as
         ``apply_period`` and reject when it is in the past.
      4. No date info present → assume open, return True.
    """
    today = today or datetime.now().date()

    recruit_status = str(
        metadata.get("rcrt_prgs_yn")
        or metadata.get("cond[rcrt_prgs_yn::EQ]")
        or ""
    ).strip()
    if recruit_status in _CLOSED_STATUSES:
        return False
    if recruit_status in _OPEN_STATUSES:
        return True

    # ── Status check ──────────────────────────────────────────────────────
    status = str(metadata.get("status") or "").strip()
    if status in _CLOSED_STATUSES:
        return False

    # ── Date check ────────────────────────────────────────────────────────
    raw = metadata.get("deadline") or metadata.get("confmdoc_expr_dt")
    d = _parse_date(raw)
    if d is not None and d < today:
        return False

    # ── Period end-date check (e.g., announcement.apply_period) ─────────────
    period_end = _extract_period_end_date(metadata)
    if period_end is not None and period_end < today:
        return False

    # ── LLM fallback decision (used only when deterministic signals are weak) ──
    llm_is_open = metadata.get("llm_deadline_is_open")
    if llm_is_open is False:
        return False
    if llm_is_open is True:
        return True

    return True
