from __future__ import annotations

from typing import Dict, Iterable


def passes_exclusion_constraint(
    special_conditions: Iterable[str] | None,
    metadata: Dict[str, object],
) -> bool:
    prfn_matr = str(
        metadata.get("prfn_matr")
        or metadata.get("preferential")
        or metadata.get("cond[prfn_matr::LIKE]")
        or ""
    ).strip()
    excludes = str(metadata.get("exclude_target") or "").strip()

    blob = " ".join([prfn_matr, excludes]).strip()
    if not blob:
        return True

    for cond in (special_conditions or []):
        if cond and cond in blob:
            return False
    return True
