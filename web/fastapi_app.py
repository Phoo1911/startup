"""
FastAPI 기반 웹 서버 + REST API
실행: uvicorn web.fastapi_app:app --reload
"""
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from models.data import UserProfile
from core.orchestrator import AgenticOrchestrator

app = FastAPI(
    title="창업지원 매칭 시스템 API",
    description="AI 기반 창업지원사업 매칭 시스템",
    version="1.0.0"
)

# 정적 파일 & 템플릿
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# 전역 Orchestrator (싱글톤)
orchestrator: Optional[AgenticOrchestrator] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global orchestrator
    try:
        Config.validate()
        orchestrator = AgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY
        )
        print("✅ Orchestrator 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        raise

# ==================== Pydantic 모델 ====================

class ProfileRequest(BaseModel):
    name: str
    age: int
    region: str
    business_stage: str
    business_field: str
    target_type: str
    is_veteran: bool = False
    is_disabled: bool = False
    additional_context: str = ""

class ChatRequest(BaseModel):
    profile: ProfileRequest
    question: str
    history: Optional[List[dict]] = []

# ==================== HTML 페이지 ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    """프로필 입력 페이지"""
    return templates.TemplateResponse("profile.html", {"request": request})

# ==================== REST API ====================

@app.post("/api/match")
async def match_profile(profile_req: ProfileRequest):
    """프로필 매칭 API"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator 미초기화")
    
    try:
        profile = UserProfile(
            name=profile_req.name,
            age=profile_req.age,
            region=profile_req.region,
            business_stage=profile_req.business_stage,
            business_field=profile_req.business_field,
            target_type=profile_req.target_type,
            is_veteran=profile_req.is_veteran,
            is_disabled=profile_req.is_disabled,
            additional_context=profile_req.additional_context
        )
        
        report = orchestrator.run(profile, top_n=10, use_cache=True)
        return JSONResponse(content=report)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """챗봇 API"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator 미초기화")
    
    try:
        profile = UserProfile(
            name=req.profile.name,
            age=req.profile.age,
            region=req.profile.region,
            business_stage=req.profile.business_stage,
            business_field=req.profile.business_field,
            target_type=req.profile.target_type,
            is_veteran=req.profile.is_veteran,
            is_disabled=req.profile.is_disabled,
            additional_context=req.profile.additional_context
        )
        
        answer = orchestrator.chatbot.chat(
            profile, 
            req.question, 
            req.history
        )
        
        return {"answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "ok",
        "orchestrator": orchestrator is not None,
        "llm_enabled": Config.LLM_API_KEY is not None
    }

# ==================== 실행 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host=Config.WEB_HOST,
        port=Config.WEB_PORT,
        reload=Config.DEBUG
    )
