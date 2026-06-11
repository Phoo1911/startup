from __future__ import annotations

import re
from typing import Any, Dict, Optional


_AGE_RANGE_RE = re.compile(r"(\d{1,2})\s*[~\-]\s*(\d{1,2})\s*세")
_AGE_UPPER_RE = re.compile(r"(\d{1,2})\s*세\s*이하")
_AGE_LOWER_RE = re.compile(r"(\d{1,2})\s*세\s*이상")


def has_explicit_age_constraint(metadata: Dict[str, Any]) -> bool:
    biz_trgt_age = str(
        metadata.get("biz_trgt_age")
        or metadata.get("cond[biz_trgt_age::LIKE]")
        or ""
    ).strip()
    if biz_trgt_age:
        return True

    text = str(metadata.get("age_limit") or "").strip()
    if not text:
        return False
    return bool(_AGE_RANGE_RE.search(text) or _AGE_UPPER_RE.search(text) or _AGE_LOWER_RE.search(text))


def passes_age_constraint(user_age: Optional[int], metadata: Dict[str, Any]) -> bool:
    if user_age is None:
        return True

    # API field priority: biz_trgt_age
    biz_trgt_age = str(
        metadata.get("biz_trgt_age")
        or metadata.get("cond[biz_trgt_age::LIKE]")
        or ""
    ).strip()
    if biz_trgt_age:
        if "만 20세 미만" in biz_trgt_age:
            return user_age < 20
        if "만 20세 이상" in biz_trgt_age and "만 39세 이하" in biz_trgt_age:
            return 20 <= user_age <= 39
        if "만 40세 이상" in biz_trgt_age:
            return user_age >= 40

    # Fallback: legacy free-text age field
    text = str(metadata.get("age_limit") or "").strip()
    if not text:
        return True
    if "청년" in text and not (19 <= user_age <= 39):
        return False
    if "중장년" in text and user_age < 40:
        return False

    m = _AGE_RANGE_RE.search(text)
    if m:
        return int(m.group(1)) <= user_age <= int(m.group(2))
    m = _AGE_UPPER_RE.search(text)
    if m and user_age > int(m.group(1)):
        return False
    m = _AGE_LOWER_RE.search(text)
    if m and user_age < int(m.group(1)):
        return False
    return True
