from __future__ import annotations

import re
from typing import Any, Dict, Optional


_STARTUP_UPPER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*년\s*이하")
_STARTUP_LOWER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*년\s*이상")
_STARTUP_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[~\-]\s*(\d+(?:\.\d+)?)\s*년")


def has_explicit_startup_constraint(metadata: Dict[str, Any]) -> bool:
    biz_enyy = str(
        metadata.get("biz_enyy")
        or metadata.get("cond[biz_enyy::LIKE]")
        or ""
    ).strip()
    if biz_enyy:
        return True

    text = str(metadata.get("startup_period") or "").strip()
    if not text:
        return False
    if "예비창업" in text:
        return True
    return bool(_STARTUP_UPPER_RE.search(text) or _STARTUP_LOWER_RE.search(text) or _STARTUP_RANGE_RE.search(text))


def passes_startup_constraint(startup_years: Optional[float], metadata: Dict[str, Any]) -> bool:
    if startup_years is None:
        return True

    # API field priority: biz_enyy
    biz_enyy = str(
        metadata.get("biz_enyy")
        or metadata.get("cond[biz_enyy::LIKE]")
        or ""
    ).strip()
    if biz_enyy:
        if "예비창업자" in biz_enyy:
            return startup_years <= 0
        for year_limit in (1, 2, 3, 5, 7):
            if f"{year_limit}년미만" in biz_enyy:
                return startup_years < year_limit
        return True

    # Fallback: legacy free-text startup_period field
    text = str(metadata.get("startup_period") or "").strip()
    if not text:
        return True
    if ("예비창업" in text or "예비" in text) and startup_years > 0:
        return False
    m = _STARTUP_UPPER_RE.search(text)
    if m and startup_years > float(m.group(1)):
        return False
    m = _STARTUP_LOWER_RE.search(text)
    if m and startup_years < float(m.group(1)):
        return False
    m = _STARTUP_RANGE_RE.search(text)
    if m and not (float(m.group(1)) <= startup_years <= float(m.group(2))):
        return False
    return True
