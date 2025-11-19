"""
텍스트 처리 유틸리티
"""
import re
from typing import Any, Dict

def clean_text(text: Any) -> str:
    """텍스트 정리"""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()

def safe_get(data: Dict, *keys, default: Any = "") -> Any:
    """안전한 딕셔너리 접근"""
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result if result is not None else default
