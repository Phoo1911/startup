"""
LLM 클라이언트
"""
from config.settings import Config

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Anthropic (선택)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# 로컬 LLM(Qwen 등)용 transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMClient:
    """OpenAI / Anthropic / 로컬 Qwen 공용 클라이언트"""

    def __init__(self, provider: str = None, api_key: str = None):
        self.provider = provider or Config.LLM_PROVIDER
        self.api_key = api_key or Config.LLM_API_KEY
        self.model = Config.LLM_MODEL

        self.client = None
        self.local_model = None
        self.local_tokenizer = None

        # 1) OpenAI
        if self.provider == "openai" and self.api_key and OPENAI_AVAILABLE:
            if Config.LLM_BASE_URL:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=Config.LLM_BASE_URL,
                )
            else:
                self.client = OpenAI(api_key=self.api_key)
            print("[LLM] OpenAI 클라이언트 초기화 완료")

        # 2) Anthropic
        elif self.provider == "anthropic" and self.api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("[LLM] Anthropic 클라이언트 초기화 완료")

        # 3) 로컬 Qwen (GPU 필요)
        elif self.provider == "local":
            model_name = Config.LOCAL_LLM_MODEL
            print(f"[LLM] 로컬 LLM 로드: {model_name}")
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )

        else:
            print("[LLM] 외부 LLM 비활성화 상태(provider=None 또는 패키지 없음)")

    # ───────────────── generate ─────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1000,
    ) -> str:
        """텍스트 생성 (provider별로 분기)"""

        try:
            # ───── 로컬 Qwen ─────
            if self.provider == "local" and self.local_model is not None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                text = self.local_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = self.local_tokenizer(
                    [text],
                    return_tensors="pt",
                ).to(self.local_model.device)

                outputs = self.local_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                )

                gen_ids = outputs[0][inputs.input_ids.shape[1] :]
                return self.local_tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                )

            # ───── OpenAI ─────
            if self.provider == "openai" and self.client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return response.choices[0].message.content

            # ───── Anthropic ─────
            if self.provider == "anthropic" and self.client:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text

            # ───── 아무것도 없는 경우 ─────
            return "[LLM 비활성화]"

        except Exception as e:
            return f"[LLM 오류: {str(e)}]"
