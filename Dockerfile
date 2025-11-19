# ============================================================
# Docker 이미지 빌드 설정
# ============================================================

FROM python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p cache reports

# 환경변수
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000 8501

# 기본 실행: Streamlit
CMD ["streamlit", "run", "web/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# FastAPI로 실행하려면:
# CMD ["uvicorn", "web.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
