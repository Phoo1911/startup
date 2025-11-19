"""
날짜 처리 유틸리티
"""
from datetime import datetime
from typing import Optional

def parse_date_flexible(date_str: str) -> Optional[datetime]:
    """유연한 날짜 파싱"""
    if not date_str:
        return None
    
    date_str = str(date_str).strip()
    formats = ['%Y%m%d', '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d']
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str[:len(fmt)], fmt)
        except:
            continue
    
    return None

def format_deadline(deadline_str: str) -> str:
    """마감일 포맷팅 (한글 strftime 에러 방지)"""
    if not deadline_str or deadline_str == '상시':
        return deadline_str
    
    deadline = parse_date_flexible(str(deadline_str).strip())
    if not deadline:
        return deadline_str
    
    # 날짜 차이 계산
    today = datetime.now().date()
    d_date = deadline.date()
    days_left = (d_date - today).days
    
    # 직접 포맷팅 (strftime 대신)
    formatted = f"{d_date.year}년 {d_date.month:02d}월 {d_date.day:02d}일"
    
    if days_left < 0:
        return f"{formatted} (마감)"
    elif days_left == 0:
        return f"{formatted} (오늘!🔥)"
    elif days_left <= 3:
        return f"{formatted} (D-{days_left}🔥)"
    elif days_left <= 7:
        return f"{formatted} (D-{days_left})"
    else:
        return formatted
