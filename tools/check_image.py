# tools/check_image_keys.py
import sys

from pathlib import Path
# streamlit_app.py 맨 위에 추가
# 프로젝트 루트 경로 (web 폴더의 한 단계 위)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from config.settings import Config
from core.llm_client import LLMClient
from core.rag_system import RAGSystem
from agents.data_collector import DataCollectionAgent

# ---- raw_data 수집 ----
service_key = Config.SERVICE_KEY
llm_client = None  # 여기서는 LLM 안 써도 됨

data_agent = DataCollectionAgent(service_key, llm_client)
raw_data = data_agent.collect_all(max_pages=1)  # 페이지 줄여서 테스트

# ---- 이미지 비슷한 키 찾기 ----
def find_image_like_keys(raw_data):
    keys = set()
    for category, items in raw_data.items():
        for item in items:
            for k in item.keys():
                if any(word in k.lower() for word in ["img", "image", "thumb", "file_url", "atfl"]):
                    keys.add((category, k))
    return keys

print("이미지 관련으로 보이는 키들:")
print(find_image_like_keys(raw_data))
