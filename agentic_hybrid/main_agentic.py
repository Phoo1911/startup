from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from agentic_hybrid.config import AgenticHybridConfig, apply_experiment_mode, load_config
from agentic_hybrid.graph import build_agentic_graph
from agentic_hybrid.state import init_state
from agentic_hybrid.nodes.intent_classifier_node import ALL_DOC_TYPES
from agentic_hybrid.vllm_client import call_llm


AUTORAG_MODE_DEFAULTS: dict[str, tuple[str, str]] = {
    "baseline": ("autorag_workspace/configs/benchmark_1.yaml", "autorag_workspace/results_baseline"),
    "autorag": ("autorag_workspace/configs/benchmark_3.yaml", "autorag_workspace/results_autorag"),
    "full": ("autorag_workspace/configs/rag_eval.yaml", "autorag_workspace/results_full"),
    "no_hyde": ("autorag_workspace/configs/rag_eval.yaml", "autorag_workspace/results_no_hyde"),
    "no_reranker": ("autorag_workspace/configs/rag_eval.yaml", "autorag_workspace/results_no_reranker"),
    "no_freshness": ("autorag_workspace/configs/rag_eval.yaml", "autorag_workspace/results_no_freshness"),
    "no_deadline": ("autorag_workspace/configs/rag_eval.yaml", "autorag_workspace/results_no_deadline"),
}


def _run_autorag_eval_for_mode(mode: str) -> None:
    mode_key = str(mode or "full").strip().lower()
    mode_conf = AUTORAG_MODE_DEFAULTS.get(mode_key)
    if not mode_conf:
        print(f"[AutoRAG] unsupported mode={mode!r}, skip")
        return

    project_root = Path(__file__).resolve().parents[1]
    config_rel, project_dir_rel = mode_conf
    config_path = project_root / config_rel
    qa_path = project_root / "autorag_workspace" / "qa" / "qa.parquet"
    corpus_path = project_root / "autorag_workspace" / "qa" / "corpus.parquet"
    project_dir = project_root / project_dir_rel

    if shutil.which("autorag") is None:
        print("[AutoRAG] 'autorag' command not found, skip")
        return
    if not config_path.exists():
        print(f"[AutoRAG] config not found: {config_path}, skip")
        return
    if not qa_path.exists() or not corpus_path.exists():
        print(
            "[AutoRAG] qa/corpus parquet not found "
            f"(qa={qa_path.exists()}, corpus={corpus_path.exists()}), skip"
        )
        return

    project_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "autorag",
        "evaluate",
        "--config",
        str(config_path),
        "--qa_data_path",
        str(qa_path),
        "--corpus_data_path",
        str(corpus_path),
        "--project_dir",
        str(project_dir),
    ]
    print(
        "[AutoRAG] evaluate start: "
        f"mode={mode_key}, config={config_path}, project_dir={project_dir}"
    )
    try:
        subprocess.run(cmd, check=True)
        print("[AutoRAG] evaluate done")
    except Exception as e:
        # Do not block online inference path when offline eval fails.
        print(f"[AutoRAG] evaluate failed, continue agentic path: {type(e).__name__}: {e}")


class SimpleLLM:
    def __init__(self, cfg: AgenticHybridConfig) -> None:
        self.cfg = cfg
        self.client = None
        self.backend = None
        self.init_error: Optional[str] = None
        self.hf_model_name: Optional[str] = None
        self.hf_provider_hint: Optional[str] = None
        self.tokenizer = None
        self.model = None
        self.encode_messages = None
        self.parse_message_from_completion_text = None
        self.last_error: Optional[str] = None
        provider = str(cfg.LLM_PROVIDER or "").strip().lower()
        if provider == "vllm":
            self.backend = "vllm"
        elif provider == "transformers":
            try:
                # Chat/instruct models (DeepSeek/Qwen/etc.) are more reliable via causal generation.
                model_name_l = cfg.LLM_MODEL_NAME.lower()
                if (
                    "deepseek-v3.2" in model_name_l
                    or "qwen/" in model_name_l
                    or "qwen-" in model_name_l
                    or "instruct" in model_name_l
                ):
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            cfg.LLM_MODEL_NAME,
                            trust_remote_code=True,
                        )
                    except Exception:
                        # Some repos ship tokenizer artifacts that fail with the
                        # fast tokenizer on older transformers/tokenizers combos.
                        # Retry with the slow tokenizer before disabling LLM use.
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            cfg.LLM_MODEL_NAME,
                            trust_remote_code=True,
                            use_fast=False,
                        )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg.LLM_MODEL_NAME,
                        trust_remote_code=True,
                        device_map="auto",
                        torch_dtype="auto",
                        attn_implementation="eager",
                    )
                    try:
                        from encoding_dsv32 import encode_messages, parse_message_from_completion_text

                        # Only use custom parser for DeepSeek-V3.2 family.
                        if "deepseek-v3.2" in model_name_l:
                            self.encode_messages = encode_messages
                            self.parse_message_from_completion_text = parse_message_from_completion_text
                        else:
                            self.encode_messages = None
                            self.parse_message_from_completion_text = None
                    except Exception:
                        self.encode_messages = None
                        self.parse_message_from_completion_text = None
                    self.backend = "transformers_causal"
                else:
                    from transformers import pipeline

                    self.client = pipeline("text-generation", model=cfg.LLM_MODEL_NAME)
                    self.backend = "transformers"
            except Exception as e:
                self.client = None
                self.backend = None
                self.init_error = f"transformers backend init failed: {type(e).__name__}: {e}"
        elif provider == "openai" and cfg.OPENAI_API_KEY:
            try:
                from openai import OpenAI

                kwargs = {"api_key": cfg.OPENAI_API_KEY}
                if cfg.OPENAI_BASE_URL:
                    kwargs["base_url"] = cfg.OPENAI_BASE_URL
                self.client = OpenAI(**kwargs)
                self.backend = "openai"
            except Exception as e:
                self.client = None
                self.backend = None
                self.init_error = f"openai backend init failed: {type(e).__name__}: {e}"
        elif provider in {"huggingface", "hf"} and cfg.HF_API_KEY:
            try:
                from huggingface_hub import InferenceClient

                # Accept router-style model names like "repo_id:provider" and normalize.
                raw_model = str(cfg.LLM_MODEL_NAME or "").strip()
                if ":" in raw_model:
                    repo_id, provider_hint = raw_model.split(":", 1)
                    self.hf_model_name = repo_id.strip()
                    self.hf_provider_hint = provider_hint.strip() or None
                else:
                    self.hf_model_name = raw_model
                    self.hf_provider_hint = None

                kwargs = {"api_key": cfg.HF_API_KEY}
                # Newer huggingface_hub supports provider=... in InferenceClient.
                if self.hf_provider_hint:
                    kwargs["provider"] = self.hf_provider_hint
                try:
                    self.client = InferenceClient(**kwargs)
                except TypeError:
                    kwargs.pop("provider", None)
                    self.client = InferenceClient(**kwargs)
                self.backend = "huggingface"
            except Exception as e:
                self.client = None
                self.backend = None
                self.init_error = f"huggingface backend init failed: {type(e).__name__}: {e}"
        elif provider == "google" and cfg.GOOGLE_API_KEY:
            try:
                from google import genai

                self.client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
                self.backend = "google"
            except Exception as e:
                self.client = None
                self.backend = None
                self.init_error = f"google backend init failed: {type(e).__name__}: {e}"
        else:
            self.init_error = f"provider={provider} is not ready (missing key or unsupported)"

    def _should_retry_error(self, error: Exception) -> bool:
        text = f"{type(error).__name__}: {error}".lower()
        retry_markers = [
            "internalservererror",
            "status': 'internal'",
            'status": "internal"',
            "error code: 500",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "rate limit",
            "429",
            "503",
        ]
        return any(marker in text for marker in retry_markers)

    def complete(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000) -> str:
        if self.backend is None:
            return ""
        self.last_error = None
        messages = []
        model_name_lower = str(getattr(self.cfg, "LLM_MODEL_NAME", "") or "").lower()

        # Gemma chat templates may reject explicit system-role messages.
        # In that case, fold the system prompt into the user prompt.
        if system_prompt and "gemma" in model_name_lower:
            merged_prompt = f"[System Instruction]\n{system_prompt}\n\n[User Request]\n{prompt}"
            messages.append({"role": "user", "content": merged_prompt})
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        max_attempts = max(1, int(os.getenv("AH_LLM_MAX_RETRIES", "3")))
        base_delay = max(0.1, float(os.getenv("AH_LLM_RETRY_BASE_DELAY", "1.0")))

        for attempt in range(1, max_attempts + 1):
            try:
                if self.backend == "vllm":
                    return call_llm(messages)
                if self.backend == "transformers_causal" and self.tokenizer is not None and self.model is not None:
                    if self.encode_messages is not None:
                        encoded_prompt = self.encode_messages(
                            messages,
                            thinking_mode="thinking",
                            drop_thinking=True,
                            add_default_bos_token=True,
                        )
                    else:
                        encoded_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )

                    inputs = self.tokenizer(encoded_prompt, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                    )
                    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
                    if self.parse_message_from_completion_text is not None:
                        try:
                            parsed: Any = self.parse_message_from_completion_text(text)
                            if isinstance(parsed, dict) and parsed.get("content"):
                                return str(parsed.get("content")).strip()
                        except Exception:
                            pass
                    return text
                if self.backend == "transformers":
                    resp = self.client(messages, max_new_tokens=max_tokens)
                    if isinstance(resp, list) and resp:
                        generated = resp[0].get("generated_text")
                        if isinstance(generated, list) and generated:
                            last = generated[-1]
                            if isinstance(last, dict):
                                return str(last.get("content", "")).strip()
                            return str(last).strip()
                        if isinstance(generated, str):
                            return generated.strip()
                        if isinstance(resp[0], dict):
                            return str(resp[0].get("text", "")).strip()
                    return ""

                if self.backend == "huggingface":
                    resp = self.client.chat.completions.create(
                        model=self.hf_model_name or self.cfg.LLM_MODEL_NAME,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=self.cfg.LLM_TEMPERATURE,
                    )
                    content = resp.choices[0].message.content
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text") or item.get("content")
                                if text:
                                    parts.append(str(text))
                            elif item:
                                parts.append(str(item))
                        return "\n".join(parts).strip()
                    return str(content or "").strip()

                if self.backend == "google":
                    config_kwargs: dict[str, Any] = {
                        "temperature": self.cfg.LLM_TEMPERATURE,
                        "max_output_tokens": max_tokens,
                    }
                    if system_prompt:
                        config_kwargs["system_instruction"] = system_prompt

                    merged_prompt = str(prompt or "").strip()
                    try:
                        from google.genai import types as genai_types

                        resp = self.client.models.generate_content(
                            model=self.cfg.LLM_MODEL_NAME,
                            contents=merged_prompt,
                            config=genai_types.GenerateContentConfig(**config_kwargs),
                        )
                    except Exception:
                        resp = self.client.models.generate_content(
                            model=self.cfg.LLM_MODEL_NAME,
                            contents=merged_prompt,
                            config=config_kwargs,
                        )

                    text = getattr(resp, "text", None)
                    finish_reasons = []
                    candidates = getattr(resp, "candidates", None) or []
                    for candidate in candidates:
                        finish_reason = getattr(candidate, "finish_reason", None)
                        if finish_reason is not None:
                            finish_reasons.append(str(finish_reason))
                    if finish_reasons:
                        print(
                            f"[LLM google] model={self.cfg.LLM_MODEL_NAME}, "
                            f"finish_reasons={finish_reasons}, max_output_tokens={max_tokens}"
                        )
                    if text:
                        return str(text).strip()

                    parts: list[str] = []
                    for candidate in candidates:
                        content = getattr(candidate, "content", None)
                        content_parts = getattr(content, "parts", None) or []
                        for part in content_parts:
                            part_text = getattr(part, "text", None)
                            if part_text:
                                parts.append(str(part_text))
                    return "\n".join(parts).strip()

                resp = self.client.chat.completions.create(
                    model=self.cfg.LLM_MODEL_NAME,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.cfg.LLM_TEMPERATURE,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                is_retryable = self._should_retry_error(e)
                if attempt < max_attempts and is_retryable:
                    delay = base_delay * (2 ** (attempt - 1))
                    print(
                        f"[LLM retry] backend={self.backend}, model={self.cfg.LLM_MODEL_NAME}, "
                        f"attempt={attempt}/{max_attempts}, wait={delay:.1f}s, err={self.last_error}"
                    )
                    time.sleep(delay)
                    continue
                print(f"[LLM complete error] backend={self.backend}, model={self.cfg.LLM_MODEL_NAME}, err={self.last_error}")
                return ""


def _to_natural_thoughts(trace: list[str]) -> list[str]:
    return _to_natural_thoughts_for_lang(trace, "ko")


def _detect_text_language(text: str) -> str:
    return "ko" if re.search(r"[\uac00-\ud7a3]", text or "") else "en"


def _thought_trace_for_display(trace: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in trace:
        text = str(line or "").strip()
        lower = text.lower()
        if not text:
            continue
        if any(token in lower for token in ["disabled", "skipped", "no docs", "no-op"]):
            continue
        if text.startswith("intent_classifier:"):
            continue
        if text.startswith("query_expansion:"):
            continue
        if text.startswith("inherit_deadline:") and "filled=0" in lower:
            continue
        if text.startswith("llm_deadline_review:") and re.search(r"reviewed=0,\s*open=0,\s*closed=0,\s*unknown=0", lower):
            continue
        cleaned.append(text)
    return cleaned


def _to_natural_thoughts_for_lang(trace: list[str], lang: str) -> list[str]:
    trace = _thought_trace_for_display(trace)
    out: list[str] = []

    def add(text: str) -> None:
        text = str(text or "").strip()
        if text and text not in out:
            out.append(text)

    for line in trace:
        detail_lower = line.lower()
        if any(token in detail_lower for token in ["disabled", "skipped", "no docs", "no-op"]):
            continue
        if line.startswith("intent_classifier:"):
            continue
        if line.startswith("query_expansion:"):
            continue
        if line.startswith("inherit_deadline:") and "filled=0" in detail_lower:
            continue
        if line.startswith("llm_deadline_review:") and re.search(r"reviewed=0,\s*open=0,\s*closed=0,\s*unknown=0", detail_lower):
            continue

        if line.startswith("retrieve:"):
            m = re.search(r"raw=(\d+),\s*filtered=(\d+)", line)
            if m:
                if lang == "ko":
                    add(f"검색 후보 {m.group(1)}건 중 문서 유형 조건에 맞는 {m.group(2)}건을 우선 추렸습니다.")
                else:
                    add(f"Retrieved {m.group(1)} candidates and kept {m.group(2)} after the initial doc-type filter.")
            continue

        if line.startswith("doc_type_router:"):
            m = re.search(r"kept\s+(\d+)/(\d+)\s+docs", line)
            if m:
                if lang == "ko":
                    add(f"선택한 데이터 유형 기준으로 후보 {m.group(1)}건을 유지했습니다.")
                else:
                    add(f"Kept {m.group(1)} documents after the strict selected-type filter.")
            continue

        if line.startswith("planner:"):
            if lang == "ko":
                parts = []
                m_age = re.search(r"age=([^,\[]+)", line)
                m_startup = re.search(r"startup_years=([^,\[]+)", line)
                m_region = re.search(r"region=([^,\[]+)", line)
                m_target = re.search(r"target_type=([^,\[]+)", line)
                if m_age and m_age.group(1) not in {"None", "null"}:
                    parts.append(f"나이 {m_age.group(1)}세")
                if m_startup and m_startup.group(1) not in {"None", "null"}:
                    try:
                        startup_years = float(m_startup.group(1))
                        if startup_years == 0:
                            parts.append("예비창업 단계")
                        else:
                            parts.append(f"창업 {startup_years:g}년차")
                    except Exception:
                        parts.append(f"창업단계 {m_startup.group(1)}")
                if m_region and m_region.group(1) not in {"None", "null"}:
                    parts.append(f"지역 {m_region.group(1)}")
                if m_target and m_target.group(1) not in {"None", "null"}:
                    parts.append(f"대상유형 {m_target.group(1)}")
                if parts:
                    add("질문에서 " + ", ".join(parts) + " 조건을 해석해 필터링 기준으로 반영했습니다.")
            continue

        if line.startswith("filter:"):
            m = re.search(r"kept=(\d+)\s+from\s+(\d+)", line)
            if m:
                if lang == "ko":
                    add(f"연령, 창업단계, 지역, 대상유형 같은 조건을 반영해 후보를 {m.group(2)}건에서 {m.group(1)}건으로 줄였습니다.")
                else:
                    add(f"Applied user constraints and reduced the pool from {m.group(2)} to {m.group(1)} documents.")
            continue

        if line.startswith("dedup:"):
            m = re.search(r"(\d+)\s*->\s*(\d+)\s*docs", line)
            if m and m.group(1) != m.group(2):
                if lang == "ko":
                    add(f"서로 비슷한 문서를 묶어 후보를 {m.group(1)}건에서 {m.group(2)}건으로 정리했습니다.")
                else:
                    add(f"Collapsed near-duplicate documents from {m.group(1)} to {m.group(2)}.")
            continue

        if line.startswith("cross_doc_enrich:"):
            m = re.search(r"enriched\s+(\d+)/(\d+)", line)
            if m and m.group(1) != "0":
                if lang == "ko":
                    add(f"연결된 공고와 사업 정보를 함께 확인해 {m.group(1)}건의 후보를 보강했습니다.")
                else:
                    add(f"Enriched {m.group(1)} documents with linked cross-document information.")
            continue

        if line.startswith("final_policy_gate:"):
            m = re.search(r"kept=(\d+)/(\d+)", line)
            if m:
                if lang == "ko":
                    add(f"최종 정책 검증을 거쳐 {m.group(2)}건 중 {m.group(1)}건만 답변 근거로 확정했습니다.")
                else:
                    add(f"Final policy validation kept {m.group(1)} of {m.group(2)} documents.")
            continue

        if line.startswith("generate:"):
            if lang == "ko":
                add("선택된 근거 문서를 바탕으로 최종 답변을 생성했습니다.")
            else:
                add("Generated the final answer from the selected evidence documents.")
            continue

        if line.startswith("revise:"):
            if lang == "ko":
                add("답변에 빠진 조건이나 근거가 없는지 마지막으로 점검했습니다.")
            else:
                add("Ran a final revision pass to check for missing constraints or evidence.")
            continue

    return out

def _llm_thoughts(llm: "SimpleLLM", question: str, trace: list[str]) -> list[str]:
    trace = _thought_trace_for_display(trace)
    if not trace:
        return []
    lang = "ko"
    prompt = (
        "아래 질문과 실행 로그를 바탕으로 사용자에게 보여줄 처리 과정을 작성하세요.\n"
        "중요: 이것은 답변 요약이 아니라 실행 단계 요약입니다.\n"
        "규칙:\n"
        "1) 실행 로그에 직접 나타난 사실만 사용하세요.\n"
        "2) 추천된 사업명, 지원내용, 자격조건, 정책 설명은 쓰지 마세요.\n"
        "3) 로그에 없는 의도, 필터, 선호도, 판단 이유를 추가하지 마세요.\n"
        "4) disabled, skipped, unused 단계는 언급하지 마세요.\n"
        "5) 실제 수행된 단계만 순서대로 연결하세요.\n"
        "6) 숫자는 반드시 로그에 있는 값만 사용하세요.\n"
        "7) filter 이후 후보 수와 final_policy_gate 이후 최종 근거 문서 수를 구분해서 쓰세요.\n"
        "8) 단계 순서가 어색해지지 않도록 검색 → 라우팅 → 질의 해석 → 필터링 → 중복 제거 → 교차 문서 보강 → 최종 정책 검증 → 답변 생성 순으로 자연스럽게 정리하세요.\n"
        "9) 로그에 값이 없거나 None인 조건은 '추출하지 못했다'라고 쓰지 말고, '질문에서 명시적으로 확인되지 않았다'처럼 중립적으로 표현하세요.\n"
        "10) filter와 dedup, final_policy_gate의 수치를 서로 모순되게 서술하지 마세요. 예를 들어 최종 근거 문서 수를 말한 뒤 다시 더 큰 후보 수를 뒤에 쓰지 마세요.\n"
        "11) 문장은 사용자에게 보여주는 처리 과정처럼 자연스럽게 쓰고, 디버그 로그처럼 쓰지 마세요.\n"
        "12) 불릿 없이 4~5문장으로 작성하세요.\n"
        "13) 반드시 한국어로만 작성하세요.\n\n"
        f"질문:\n{question}\n\n"
        f"실행 로그:\n" + "\n".join(f"- {t}" for t in trace)
    )
    text = llm.complete(
        prompt,
        system_prompt=(
            "당신은 검색 파이프라인의 처리 과정을 사용자에게 자연스럽게 설명하는 시스템입니다. "
            "절대로 답변 내용이나 추천 이유를 설명하지 말고, 실제 실행된 단계만 요약하세요. "
            "실행 로그에 직접 나타난 사실만 사용하세요. "
            "없는 단계, 이유, 조건을 지어내지 마세요. "
            "단계 순서가 어색하지 않도록 정리하고, 후보 수와 최종 근거 문서 수를 구분하세요. "
            "조건이 없을 때는 실패처럼 쓰지 말고 중립적으로 표현하세요. "
            "반드시 한국어로만 작성하세요."
        ),
        max_tokens=900,
    )
    if not text:
        return []

    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(("-", "*", "•", "?")):
            s = s[1:].strip()
        if len(s) >= 3 and s[:2].isdigit() and s[2:3] in {".", ")"}:
            s = s[3:].strip()
        if s:
            lines.append(s)
    return lines[:10]

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Agentic Hybrid LangGraph RAG")
    parser.add_argument("-q", "--question", required=True, help="User query")
    parser.add_argument(
        "--mode",
        choices=["baseline", "autorag", "full", "no_hyde", "no_reranker", "no_freshness", "no_deadline"],
        default="full",
    )
    parser.add_argument("--top-k-final", type=int, default=None)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument(
        "--autorag-link",
        choices=["on", "off"],
        default="on",
        help="on: run AutoRAG evaluate with the same --mode before agentic inference",
    )
    parser.add_argument(
        "--doc-types",
        type=str,
        default="",
        help=f"Optional UI override, comma-separated doc types. allowed={','.join(ALL_DOC_TYPES)}",
    )
    args = parser.parse_args()

    if args.autorag_link == "on":
        _run_autorag_eval_for_mode(args.mode)

    cfg = apply_experiment_mode(load_config(), args.mode)
    if args.top_k_final is not None:
        cfg = replace(cfg, TOP_K_FINAL=args.top_k_final)

    llm = SimpleLLM(cfg)
    app = build_agentic_graph(cfg, llm)
    selected_doc_types = [
        t.strip().lower()
        for t in (args.doc_types or "").split(",")
        if t.strip().lower() in ALL_DOC_TYPES
    ]
    result = app.invoke(init_state(args.question, selected_doc_types=selected_doc_types))

    print("\n[Final Answer]")
    print(result.get("answer", ""))

    if not args.no_trace:
        print("\n[Reasoning Trace]")
        trace_lines = result.get("reasoning_trace", [])
        for line in trace_lines:
            print(f"- {line}")

            thought_lines = _llm_thoughts(llm, args.question, trace_lines)
        if not thought_lines:
            thought_lines = _to_natural_thoughts_for_lang(trace_lines, _detect_text_language(args.question))
        if thought_lines:
            print("\n[Thought]")
            for line in thought_lines:
                print(f"- {line}")

    if "final_docs" in result:
        final_docs = result.get("final_docs") or []
    else:
        final_docs = result.get("filtered_docs") or []
    if final_docs:
        print("\n[Final Docs]")
        for i, doc in enumerate(final_docs[: cfg.TOP_K_FINAL], start=1):
            md = doc.get("metadata", {}) or {}
            print(f"{i}. {doc.get('title','')} | {md.get('type','')} | {md.get('deadline','')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
