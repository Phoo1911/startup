from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_INPUT = Path("autorag_workspace/results_agent/predictions.parquet")
DEFAULT_OUTPUT = Path("autorag_workspace/results_agent/hallucination_metrics.csv")
DEFAULT_JUDGE_BACKEND = "openai"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-8B"

_HF_TOKENIZER = None
_HF_MODEL = None
_SEM_MODEL = None


def _as_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s[0] in "[{(" and s[-1] in "]})":
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(s)
                except Exception:
                    pass
        return s
    return value


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9?-?]+", (text or "").lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    if len(tokens) < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_score(reference: str, candidate: str, max_n: int = 4) -> Optional[float]:
    ref = _tokenize(reference)
    cand = _tokenize(candidate)
    if not ref or not cand:
        return None
    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_counts = _ngram_counts(ref, n)
        cand_counts = _ngram_counts(cand, n)
        total = sum(cand_counts.values())
        if total == 0:
            precisions.append(0.0)
            continue
        clipped = 0
        for gram, count in cand_counts.items():
            clipped += min(count, ref_counts.get(gram, 0))
        precisions.append(clipped / total)
    smooth = [max(p, 1e-9) for p in precisions]
    geo_mean = math.exp(sum(math.log(p) for p in smooth) / max_n)
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - (len(ref) / len(cand)))
    return bp * geo_mean


def _overlap_f1(ref_tokens: List[str], cand_tokens: List[str], n: int) -> Optional[float]:
    ref_counts = _ngram_counts(ref_tokens, n)
    cand_counts = _ngram_counts(cand_tokens, n)
    if not ref_counts or not cand_counts:
        return None
    overlap = sum(min(count, cand_counts.get(gram, 0)) for gram, count in ref_counts.items())
    precision = overlap / max(sum(cand_counts.values()), 1)
    recall = overlap / max(sum(ref_counts.values()), 1)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def _rouge_scores(reference: str, candidate: str) -> Dict[str, Optional[float]]:
    ref = _tokenize(reference)
    cand = _tokenize(candidate)
    rouge1 = _overlap_f1(ref, cand, 1)
    rouge2 = _overlap_f1(ref, cand, 2)
    if not ref or not cand:
        rougel = None
    else:
        lcs = _lcs_length(ref, cand)
        p = lcs / len(cand) if cand else 0.0
        r = lcs / len(ref) if ref else 0.0
        rougel = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"rouge1_f1": rouge1, "rouge2_f1": rouge2, "rougeL_f1": rougel}


def _meteor_score(reference: str, candidate: str) -> Optional[float]:
    ref = _tokenize(reference)
    cand = _tokenize(candidate)
    if not ref or not cand:
        return None
    ref_counts = Counter(ref)
    cand_counts = Counter(cand)
    matches = sum(min(count, cand_counts.get(tok, 0)) for tok, count in ref_counts.items())
    precision = matches / len(cand)
    recall = matches / len(ref)
    return (10 * precision * recall / (recall + 9 * precision)) if (precision + recall) else 0.0


def _load_semantic_model(model_name: str):
    global _SEM_MODEL
    if _SEM_MODEL is not None:
        return _SEM_MODEL
    from sentence_transformers import SentenceTransformer
    _SEM_MODEL = SentenceTransformer(model_name)
    return _SEM_MODEL


def _sem_score(reference: str, candidate: str, model_name: str) -> Optional[float]:
    if not reference or not candidate:
        return None
    try:
        model = _load_semantic_model(model_name)
        embeddings = model.encode([reference, candidate], normalize_embeddings=True)
        return float((embeddings[0] * embeddings[1]).sum())
    except Exception:
        return None


def _docs_to_context(retrieved_docs: Any, max_chars: int = 2500, max_docs: int = 5) -> str:
    obj = _as_obj(retrieved_docs)
    texts: List[str] = []
    if isinstance(obj, list):
        for item in obj[:max_docs]:
            if isinstance(item, dict):
                txt = str(item.get("text") or item.get("contents") or item.get("content") or item.get("summary") or "").strip()
                title = str(item.get("title") or "").strip()
                txt = _truncate_text(txt, 400)
                if title and txt:
                    texts.append(f"[{title}] {txt}")
                elif txt:
                    texts.append(txt)
            elif item is not None:
                s = str(item).strip()
                if s:
                    texts.append(s)
    elif isinstance(obj, dict):
        txt = str(obj.get("text") or obj.get("contents") or obj.get("content") or obj.get("summary") or "").strip()
        if txt:
            texts.append(txt)
    elif obj is not None:
        s = str(obj).strip()
        if s:
            texts.append(s)
    context = "\n\n".join(texts).strip()
    return context[:max_chars]


def _truncate_text(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n...[truncated]"


def _judge_prompt(question: str, answer: str, docs_context: str, ground_truth: str) -> str:
    return (
        "You are a strict evaluator for RAG question answering.\n"
        "You must output exactly one JSON object and nothing else.\n"
        "Do not include markdown, code fences, explanations, or extra text.\n"
        "Do not mention memory, tokens, truncation, or system limitations.\n"
        "If the answer is incomplete, score it using the rubric instead of describing an error.\n"
        "You must separately score correctness and faithfulness.\n\n"
        "Definitions:\n"
        "1) Correctness compares the answer against the ground truth answer.\n"
        "2) Faithfulness compares the answer against the retrieved documents only.\n\n"
        "Scoring rubric:\n"
        "Correctness:\n"
        "-1: answer contradicts the ground truth or states a materially wrong fact\n"
        " 0: answer fails to provide the requested information, is mostly irrelevant, or misses nearly all key facts\n"
        " 1: answer is partially correct but incomplete, vague, or missing important key facts\n"
        " 2: answer fully matches the key facts in the ground truth without material error\n\n"
        "Faithfulness:\n"
        "-1: answer includes hallucinated or context-contradicted claims\n"
        " 0: answer may be plausible but one or more important claims are not supported by the retrieved documents\n"
        " 1: every material claim in the answer is supported by the retrieved documents\n\n"
        "Evaluation procedure:\n"
        "Step 1. Extract the key information nuggets from the ground truth.\n"
        "Step 2. Split the answer into material claims or sentences.\n"
        "Step 3. For each claim, determine whether it is supported by the retrieved documents.\n"
        "Step 4. List missing key nuggets, unsupported claims, and contradicted claims.\n"
        "Step 5. Assign final correctness and faithfulness scores using the rubric above.\n\n"
        "Return strict JSON only with this schema:\n"
        '{'
        '"correctness": -1|0|1|2, '
        '"faithfulness": -1|0|1, '
        '"missing_nuggets": ["..."], '
        '"unsupported_claims": ["..."], '
        '"contradicted_claims": ["..."], '
        '"reason": "short summary"'
        '}\n\n'
        f"Question:\n{_truncate_text(question, 500)}\n\n"
        f"Ground Truth:\n{_truncate_text(ground_truth, 1200) if ground_truth else '[EMPTY]'}\n\n"
        f"Generated Answer:\n{_truncate_text(answer, 1200)}\n\n"
        f"Retrieved Documents:\n{_truncate_text(docs_context, 2500) if docs_context else '[EMPTY]'}\n"
    )


def _g_eval_prompt(question: str, answer: str, docs_context: str) -> str:
    return (
        "You are an evaluator for answer quality.\n"
        "Score the answer on a 1-5 scale for coherence, consistency, fluency, and relevance.\n"
        "Use the question and retrieved documents as context.\n"
        "Return strict JSON only with schema:\n"
        '{"coherence": 1, "consistency": 1, "fluency": 1, "relevance": 1, "reason": "short reason"}\n\n'
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Retrieved Documents:\n{docs_context if docs_context else '[EMPTY]'}\n"
    )


def _call_openai_judge(prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
    from openai import OpenAI
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    return _parse_judge_json((resp.choices[0].message.content or "").strip())


def _load_transformers_judge(model_name: str):
    global _HF_TOKENIZER, _HF_MODEL
    if _HF_TOKENIZER is not None and _HF_MODEL is not None:
        return _HF_TOKENIZER, _HF_MODEL
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _HF_MODEL = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    return _HF_TOKENIZER, _HF_MODEL


def _extract_first_json_object(text: str) -> Optional[str]:
    s = _strip_think_and_fences(text)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:idx + 1]
    return None


def _call_transformers_judge(prompt: str, model_name: str, max_new_tokens: int = 512) -> Dict[str, Any]:
    tokenizer, model = _load_transformers_judge(model_name)
    messages = [
        {"role": "system", "content": "Return exactly one valid JSON object and no other text."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return _parse_judge_json(raw)


def _strip_think_and_fences(text: str) -> str:
    s = text or ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = s.replace("```json", "").replace("```", "")
    return s.strip()


def _parse_judge_json(raw_text: str) -> Dict[str, Any]:
    cleaned = _strip_think_and_fences(raw_text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    json_candidate = _extract_first_json_object(cleaned)
    if json_candidate:
        try:
            obj = json.loads(json_candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    matches = re.findall(r"\{[\s\S]*?\}", cleaned)
    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    try:
        obj = ast.literal_eval(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    corr_m = re.search(r"['\"]?correctness['\"]?\s*[:=]\s*(-?1|0|1|2)\b", cleaned, flags=re.IGNORECASE)
    faith_m = re.search(r"['\"]?faithfulness['\"]?\s*[:=]\s*(-?1|0|1)\b", cleaned, flags=re.IGNORECASE)
    miss_m = re.search(r"missing[_\s-]?nuggets?\s*[:=]\s*(\[[\s\S]*?\])", cleaned, flags=re.IGNORECASE)
    unsup_m = re.search(r"unsupported[_\s-]?claims?\s*[:=]\s*(\[[\s\S]*?\])", cleaned, flags=re.IGNORECASE)
    contra_m = re.search(r"contradicted[_\s-]?claims?\s*[:=]\s*(\[[\s\S]*?\])", cleaned, flags=re.IGNORECASE)
    coh_m = re.search(r"['\"]?coherence['\"]?\s*[:=]\s*([1-5])\b", cleaned, flags=re.IGNORECASE)
    con_m = re.search(r"['\"]?consistency['\"]?\s*[:=]\s*([1-5])\b", cleaned, flags=re.IGNORECASE)
    flu_m = re.search(r"['\"]?fluency['\"]?\s*[:=]\s*([1-5])\b", cleaned, flags=re.IGNORECASE)
    rel_m = re.search(r"['\"]?relevance['\"]?\s*[:=]\s*([1-5])\b", cleaned, flags=re.IGNORECASE)
    if corr_m or faith_m or coh_m or con_m or flu_m or rel_m:
        def _parse_list_match(match: Optional[re.Match[str]]) -> List[str]:
            if not match:
                return []
            try:
                parsed = ast.literal_eval(match.group(1))
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
            return []
        return {
            "correctness": int(corr_m.group(1)) if corr_m else None,
            "faithfulness": int(faith_m.group(1)) if faith_m else None,
            "missing_nuggets": _parse_list_match(miss_m),
            "unsupported_claims": _parse_list_match(unsup_m),
            "contradicted_claims": _parse_list_match(contra_m),
            "coherence": int(coh_m.group(1)) if coh_m else None,
            "consistency": int(con_m.group(1)) if con_m else None,
            "fluency": int(flu_m.group(1)) if flu_m else None,
            "relevance": int(rel_m.group(1)) if rel_m else None,
            "reason": "Parsed from non-JSON text",
        }
    return {"reason": f"Unparseable judge output: {cleaned[:300]}"}


def _coerce_score(value: Any, allowed: set[int]) -> Optional[int]:
    try:
        if value is None:
            return None
        iv = int(value)
        if iv in allowed:
            return iv
    except Exception:
        pass
    return None


def _coerce_str_list(value: Any) -> List[str]:
    obj = _as_obj(value)
    if obj is None:
        return []
    if isinstance(obj, dict):
        out: List[str] = []
        for item in obj.values():
            out.extend(_coerce_str_list(item))
        return out
    if isinstance(obj, (list, tuple, set)):
        out: List[str] = []
        for item in obj:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(obj).strip()
    return [text] if text else []


def _maybe_call_judge(prompt: str, backend: str, model: str, max_new_tokens: int) -> Dict[str, Any]:
    return _call_transformers_judge(prompt, model, max_new_tokens) if backend == "transformers" else _call_openai_judge(prompt, model)


def evaluate_hallucination(input_path: Path, output_path: Path, judge_model: str, judge_backend: str = DEFAULT_JUDGE_BACKEND, max_new_tokens: int = 512, debug_raw_output: bool = False, semantic_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", enable_sem_score: bool = False, enable_bert_score: bool = False, enable_g_eval: bool = False) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_parquet(input_path)
    required = {"question", "answer", "retrieved_docs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    total = len(df)
    print(f"[INFO] Loaded {total} rows from {input_path}")
    print(f"[INFO] Judge backend: {judge_backend}")
    print(f"[INFO] Judge model: {judge_model}")

    row_results: List[Dict[str, Any]] = []
    bert_candidates: List[str] = []
    bert_references: List[str] = []

    for i, row in df.iterrows():
        idx = i + 1
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        reference = str(row.get("concise_generation_gt") or row.get("generation_gt") or "")
        docs_context = _docs_to_context(row.get("retrieved_docs"))

        bleu = _bleu_score(reference, answer)
        rouge = _rouge_scores(reference, answer)
        meteor = _meteor_score(reference, answer)
        sem_score = _sem_score(reference, answer, semantic_model) if enable_sem_score else None

        if enable_bert_score:
            bert_candidates.append(answer)
            bert_references.append(reference)

        try:
            judge_out = _maybe_call_judge(
                _judge_prompt(question, answer, docs_context, reference),
                judge_backend,
                judge_model,
                max_new_tokens,
            )
            correctness = _coerce_score(judge_out.get("correctness"), {-1, 0, 1, 2})
            faithfulness = _coerce_score(judge_out.get("faithfulness"), {-1, 0, 1})
            reason = str(judge_out.get("reason") or "")
            missing_nuggets = _coerce_str_list(judge_out.get("missing_nuggets"))
            unsupported_claims = _coerce_str_list(judge_out.get("unsupported_claims"))
            contradicted_claims = _coerce_str_list(judge_out.get("contradicted_claims"))
            if debug_raw_output:
                print(f"[DBG] row={idx} parsed={judge_out}")
        except Exception as exc:
            correctness = None
            faithfulness = None
            reason = f"JudgeError: {type(exc).__name__}: {exc}"
            missing_nuggets = []
            unsupported_claims = []
            contradicted_claims = []
            print(f"[ERR] row={idx} JudgeError: {type(exc).__name__}: {exc}")

        coherence = consistency = fluency = relevance = None
        if enable_g_eval:
            try:
                g_out = _maybe_call_judge(_g_eval_prompt(question, answer, docs_context), judge_backend, judge_model, max_new_tokens)
                coherence = _coerce_score(g_out.get("coherence"), {1, 2, 3, 4, 5})
                consistency = _coerce_score(g_out.get("consistency"), {1, 2, 3, 4, 5})
                fluency = _coerce_score(g_out.get("fluency"), {1, 2, 3, 4, 5})
                relevance = _coerce_score(g_out.get("relevance"), {1, 2, 3, 4, 5})
            except Exception:
                pass

        row_results.append({
            "mode": row.get("mode", "all"),
            "correctness": correctness,
            "faithfulness": faithfulness,
            "reason": reason,
            "missing_nuggets": missing_nuggets,
            "unsupported_claims": unsupported_claims,
            "contradicted_claims": contradicted_claims,
            "missing_nuggets_count": len(missing_nuggets),
            "unsupported_claims_count": len(unsupported_claims),
            "contradicted_claims_count": len(contradicted_claims),
            "bleu": bleu,
            "rouge1_f1": rouge.get("rouge1_f1"),
            "rouge2_f1": rouge.get("rouge2_f1"),
            "rougeL_f1": rouge.get("rougeL_f1"),
            "meteor": meteor,
            "sem_score": sem_score,
            "coherence": coherence,
            "consistency": consistency,
            "fluency": fluency,
            "relevance": relevance,
        })
        print(f"[RUN] {idx}/{total} -> correctness={correctness} faithfulness={faithfulness}")

    res_df = pd.DataFrame(row_results)

    if enable_bert_score:
        try:
            from bert_score import score as bert_score
            p, r, f1 = bert_score(bert_candidates, bert_references, lang="ko", verbose=False)
            res_df["bert_precision"] = [float(x) for x in p]
            res_df["bert_recall"] = [float(x) for x in r]
            res_df["bert_f1"] = [float(x) for x in f1]
        except Exception:
            res_df["bert_precision"] = None
            res_df["bert_recall"] = None
            res_df["bert_f1"] = None

    def _avg(frame: pd.DataFrame, col: str) -> Optional[float]:
        if col not in frame.columns:
            return None
        s = frame[col].dropna()
        return float(s.mean()) if len(s) else None

    def _metrics(frame: pd.DataFrame, group_name: str) -> Dict[str, Any]:
        n = len(frame)
        valid_f = frame["faithfulness"].notna().sum()
        valid_c = frame["correctness"].notna().sum()
        hall = int((frame["faithfulness"] == -1).sum())
        grounded = int((frame["faithfulness"] == 1).sum())
        unknown = int(frame["faithfulness"].isna().sum())
        return {
            "mode": group_name,
            "hallucination_rate": (hall / n if n else 0.0),
            "grounded_ratio": (grounded / n if n else 0.0),
            "unknown_ratio": (unknown / n if n else 0.0),
            "correctness_avg": (float(frame["correctness"].dropna().mean()) if valid_c else 0.0),
            "faithfulness_avg": (float(frame["faithfulness"].dropna().mean()) if valid_f else 0.0),
            "missing_nuggets_avg": _avg(frame, "missing_nuggets_count"),
            "unsupported_claims_avg": _avg(frame, "unsupported_claims_count"),
            "contradicted_claims_avg": _avg(frame, "contradicted_claims_count"),
            "bleu_avg": _avg(frame, "bleu"),
            "rouge1_f1_avg": _avg(frame, "rouge1_f1"),
            "rouge2_f1_avg": _avg(frame, "rouge2_f1"),
            "rougeL_f1_avg": _avg(frame, "rougeL_f1"),
            "meteor_avg": _avg(frame, "meteor"),
            "sem_score_avg": _avg(frame, "sem_score"),
            "bert_precision_avg": _avg(frame, "bert_precision"),
            "bert_recall_avg": _avg(frame, "bert_recall"),
            "bert_f1_avg": _avg(frame, "bert_f1"),
            "g_eval_coherence_avg": _avg(frame, "coherence"),
            "g_eval_consistency_avg": _avg(frame, "consistency"),
            "g_eval_fluency_avg": _avg(frame, "fluency"),
            "g_eval_relevance_avg": _avg(frame, "relevance"),
            "n_samples": n,
        }

    rows = [_metrics(res_df, "ALL")]
    if "mode" in df.columns:
        for mode_name in sorted({str(x) for x in df["mode"].dropna().tolist()}):
            rows.append(_metrics(res_df[res_df["mode"].astype(str) == mode_name], mode_name))

    metrics_df = pd.DataFrame(rows)
    detailed_path = output_path.with_name(f"{output_path.stem}_detailed.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    res_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    print("\nHallucination Summary")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {output_path}")
    print(f"Saved detailed rows: {detailed_path}")
    return metrics_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hallucination and generation quality")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-backend", choices=["openai", "transformers"], default=DEFAULT_JUDGE_BACKEND)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug-raw-output", action="store_true")
    parser.add_argument("--semantic-model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--enable-sem-score", action="store_true")
    parser.add_argument("--enable-bert-score", action="store_true")
    parser.add_argument("--enable-g-eval", action="store_true")
    args = parser.parse_args()

    evaluate_hallucination(
        input_path=args.input_path,
        output_path=args.output_path,
        judge_model=args.judge_model,
        judge_backend=args.judge_backend,
        max_new_tokens=args.max_new_tokens,
        debug_raw_output=args.debug_raw_output,
        semantic_model=args.semantic_model,
        enable_sem_score=args.enable_sem_score,
        enable_bert_score=args.enable_bert_score,
        enable_g_eval=args.enable_g_eval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
