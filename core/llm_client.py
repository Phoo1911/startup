"""
LLM 클라이언트
"""
from config.settings import Config

# LLM 라이브러리 임포트 시도
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class LLMClient:
    """LLM 클라이언트"""
    
    def __init__(self, provider: str = None, api_key: str = None):
        self.provider = provider or Config.LLM_PROVIDER
        self.api_key = api_key or Config.LLM_API_KEY
        self.model = Config.LLM_MODEL
        
        if self.provider == "openai" and self.api_key and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
        elif self.provider == "anthropic" and self.api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, system_prompt: str = None, max_tokens: int = 1000) -> str:
        """텍스트 생성"""
        try:
            if self.provider == "openai" and OPENAI_AVAILABLE:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            
            else:
                return "[LLM 비활성화]"
        
        except Exception as e:
            return f"[LLM 오류: {str(e)}]"
